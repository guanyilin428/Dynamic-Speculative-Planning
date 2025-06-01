import asyncio
import time
import nltk
import torch
import os
import random
import json
import autogen
import argparse
from datetime import datetime
from agents.tool_agents_sp import DirectAgent, ReactAgent, CoTAgent, MultiAgent
from async_online_utils import OnlineLearningExecutor, SharedState
from predictor import DistilBERTValueFunction
from datasets import load_dataset

from collections import defaultdict
from transformers import AutoModel, AutoTokenizer


from util import Logger, cancel
import config
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


def judge_to_be_true(s, t):
    try:
        approximation_function_name = s.split("[")[0].strip()
        target_function_name = t.split("[")[0].strip()

        approximation_function_arg = s[s.index("[") : s.index("]")].strip()
        target_function_arg = t[t.index("[") : t.index("]")].strip()

        def token_edit_levenstein_similarity_normalized(
            text1: str, text2: str
        ) -> float:
            """
            Compute the normalized levenstein distance between two texts.
            """
            return 1 - nltk.edit_distance(text1, text2) / max(len(text1), len(text2))

        if approximation_function_name == target_function_name:
            if (
                token_edit_levenstein_similarity_normalized(
                    approximation_function_arg, target_function_arg
                )
                > 0.5
            ):
                return True

        return False
    except:
        if s == t:
            return True
        else:
            return False

def concurrent_calls():
    tasks = asyncio.all_tasks()
    pending_tasks = [t for t in tasks if not t.done() and not t.cancelled()]
    return len(pending_tasks)


### functions that can be customized
def parse_user_input(user_input, s, t):
    if user_input == "approximation answer":
        return s[0]
    elif user_input == "target answer":
        return t[0][1][0]
    elif user_input == "":
        return t[0][1][0]
    else:
        return user_input


### functions that can be customized
def interaction_function(s, t, logger, collector, previous_steps):

    cur_time = time.time()
    timestamp = datetime.fromtimestamp(cur_time)
    source = "Target"
    step = t[0][0] + len(previous_steps) + 1
    desc = t[0][1][0]

    collector.record_step(timestamp, source, step, desc)
    try:
        config.TOTAL_APPROXIMATION_CALLS += 1
        if judge_to_be_true(s[0], t[0][1][0]) and (not t[0][1][1] or t[0][1][0].lower() == 'terminate'):
            config.TOTAL_CORRECT_APPROXIMATION_CALLS += 1
            logger.log(f"Agree: Step {step}")
            logger.log(f"approximation: {s[0]}")
            logger.log(f"target: {t[0][1][0]}")
        else:
            logger.log(f"Correcting: Step {step}")
            logger.log(f"approximation: {s[0]}")
            logger.log(f"target: {t[0][1][0]}")
        user_input = ""
        user_input = parse_user_input(user_input, s, t)
    except KeyboardInterrupt as e:
        user_input = input("")

        user_input = parse_user_input(user_input, s, t)
        if not judge_to_be_true(user_input, str(t[0][1][0])):
            print(
                f"Since you think the action should be {user_input} ... we will follow your suggestion :)"
            )
        print("-------------------")

    return user_input


def ordinal(n):
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


def simulate_within_T_interaction(
    sas, tas, flatten_tas, printed_ids, logger, tar_assistant, collector, previous_steps
):
    ## conduct on-time printing out
    tas_ids = [t[0] for t in flatten_tas]
    flatten_printed_ids = []
    for ids in printed_ids:
        if ids:
            flatten_printed_ids += ids
    tas_length = len(tas_ids)
    for l in range(0, tas_length + 1):
        if not (
            tas_ids[:l] == list(range(len(flatten_tas)))[:l]
            and len(sas) >= len(flatten_tas[:l])
        ):
            tas_length = l - 1
            break

    # if the printed ids contain wrong results
    contain_wrong_result = False
    if tas_length > 0:
        for printed_id in flatten_printed_ids:
            if not judge_to_be_true(sas[printed_id][0], tas[printed_id][0][1][0]):
                contain_wrong_result = True

    if tas_length > 0 and not contain_wrong_result:
        tas_ids = tas_ids[:tas_length]
        to_print_ids = list(set(tas_ids) - set(flatten_printed_ids))
        for order_id, to_print_id in enumerate(to_print_ids):
            if order_id > 0:
                if not judge_to_be_true(
                    sas[to_print_ids[order_id - 1]][0],
                    tas[to_print_ids[order_id - 1]][0][1][0],
                ):
                    break
            printed_ids[to_print_id].append(to_print_id)
            user_input = interaction_function(
                sas[to_print_id], tas[to_print_id], logger, collector, previous_steps
            )
            if (
                str(user_input) == str(tas[to_print_id][0][1][0])
                and user_input.lower() != "terminate"
            ):
                continue
            elif (
                str(user_input) == str(tas[to_print_id][0][1][0])
                and user_input.lower() == "terminate"
            ):
                return printed_ids, tas
            else:
                change_one_tas_position = list(tas[to_print_id][0][0])
                tar_assistant.execute(user_input)
                change_one_tas_position[1] = [
                    user_input,
                    tar_assistant.current_observation,
                ]
                tas[to_print_id][0] = change_one_tas_position
                return printed_ids, tas

    ## finish printing out
    return printed_ids, tas


def simulate_leftover_interaction(
    collector, sas, tas, flatten_tas, printed_ids, logger, tar_assistant, previous_steps
):
    # when sas is slower than tas
    # we need to print out the extra here
    flatten_printed_ids = []
    for ids in printed_ids:
        if ids:
            flatten_printed_ids += ids
    print_leftover = True
    if len(sas) > len(flatten_printed_ids) and len(flatten_tas) > len(
        flatten_printed_ids
    ):
        for printed_id in flatten_printed_ids:
            if not judge_to_be_true(sas[printed_id][0], tas[printed_id][0][1][0]):
                print_leftover = False
    else:
        print_leftover = False

    # if the printed ids contain wrong results
    contain_wrong_result = False
    for printed_id in flatten_printed_ids:
        if not judge_to_be_true(sas[printed_id][0], tas[printed_id][0][1][0]):
            contain_wrong_result = True

    if print_leftover and not contain_wrong_result:
        for print_id in range(len(flatten_printed_ids), min(len(sas), len(tas))):
            if tas[print_id]:
                user_input = interaction_function(sas[print_id], tas[print_id], logger, collector, previous_steps)
                printed_ids[print_id].append(print_id)
                if str(user_input) == str(tas[print_id][0][1][0]):
                    continue
                else:
                    change_one_tas_position = list(tas[print_id][0][0])
                    tar_assistant.execute(user_input)
                    change_one_tas_position[1] = [
                        user_input,
                        tar_assistant.current_observation,
                    ]
                    tas[print_id][0] = change_one_tas_position
            else:
                break
            if not judge_to_be_true(sas[print_id][0], tas[print_id][0][1][0]):
                break
            if user_input.lower() == "terminate":
                break

    return printed_ids, tas


async def A_generate(assistant, total_step_number, approximation_logger, collector, start):

    # utilize the scratch pad
    action, finished = await assistant.direct_act(total_step_number, config, approximation_logger)
    
    end = time.time()
    a_time = round(end-start, 2)
    approximation_logger.log(f'Approximation: Step {total_step_number+1} -Action {action} -Finished {finished} -time {str(a_time)} -token {config.APPROX_NORMAL_GENERATION[total_step_number]}')
    config.APPROX_NROMAL_TIME[total_step_number+1] = a_time

    timestamp = datetime.fromtimestamp(end)
    source = "Approximation"
    step = total_step_number+1
    desc = action
    collector.record_step(timestamp, source, step, desc)

    # find action
    if not finished:
        assistant.execute(action)
        observation = assistant.current_observation
    else:
        observation = "terminate"

    return action, observation


async def T_generate(
    assistant,
    prediction_task,
    total_step_number,
    tas,
    sas,
    previous_steps,
    target_tasks,
    printed_ids,
    current_step,
    logger,
    target_logger,
    collector,
    mismatch_state,
    start
):

    # add approximation result to the prompt
    scratchpad = ""
    scratchpad = assistant.create_scratchpad(scratchpad, previous_steps + sas)
    # call agent to generate the response
    try:
        action, finished = await assistant.think_and_act(
            scratchpad, total_step_number, config, target_logger
        )
    except asyncio.CancelledError:
        return tas, printed_ids

    in_step_number = total_step_number - current_step
    tas[in_step_number] = [[in_step_number, (action, finished)]]

    # cancel next target_tasks after which we know is incorrect
    # but there maybe unknown result before this point, so we don't kill all processes
    flatten_tas = []
    for t in tas:
        if t:
            flatten_tas += t
    flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
    printed_ids, tas = simulate_within_T_interaction(
        sas, tas, flatten_tas, printed_ids, logger, assistant, collector, previous_steps
    )

    # if the target result is terminate, we break the loop
    flatten_ids = [t[0] for t in flatten_tas]
    if flatten_ids == list(range(len(flatten_ids))):
        for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
            if t[0] == step_number and t[1][1]:  # terminate
                # throw Exception here to halt everything
                if not mismatch_state.mismatch_detected.is_set():
                    mismatch_state.mismatch_step_id = t[0]
                    mismatch_state.mismatch_detected.set()
                end = time.time()
                t_time = round(end-start, 2)
                target_logger.log(f"Intermediate Target Step {total_step_number+1} -Action {action} -Finished {finished} -gen {config.TARGET_NORMAL_GENERATION[total_step_number+1]} -prompt {config.TARGET_NORMAL_PROMPT[total_step_number+1]}")
                target_logger.log(f"Target Step {total_step_number+1}: -time {t_time}")
                config.TARGET_NORMAL_TIME[total_step_number+1] = t_time
                raise Exception("terminate the whole process!")

    # if it is a wrong result
    # we break out the target processes and cancel processes that comes after it
    flatten_tas = []
    for t in tas:
        if t:
            flatten_tas += t
    flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
    for ta in flatten_tas:
        if len(sas) > ta[0]:
            # if approximation action != target action
            if not sas[ta[0]][0] == ta[1][0]:
                if not mismatch_state.mismatch_detected.is_set():
                    mismatch_state.mismatch_step_id = ta[0]
                    mismatch_state.mismatch_detected.set()
                end = time.time()
                t_time = round(end-start, 2)
                target_logger.log(f"Intermediate Target Step {total_step_number+1} -Action {action} -Finished {finished} -gen {config.TARGET_NORMAL_GENERATION[total_step_number+1]} -prompt {config.TARGET_NORMAL_PROMPT[total_step_number+1]}")
                target_logger.log(f"Target Step {total_step_number+1}: -time {t_time}")
                config.TARGET_NORMAL_TIME[total_step_number+1] = t_time
                
                pending_approximation_tasks = [
                    t
                    for t in asyncio.all_tasks()
                    if not t.cancelled()
                    and not t.done()
                    and t not in target_tasks
                    and t.get_name().startswith("approximation")
                ]
                for pending_approximation_task in pending_approximation_tasks:
                    await cancel(pending_approximation_task)
                raise Exception(
                    f"approximation error happen in step {total_step_number} for current step {current_step}, the target id is {ta[0]}"
                )
    if config.ENABLE_PRED: 
        k = await prediction_task
    else: 
        k = args.k

    end = time.time()
    t_time = round(end-start, 2)
    target_logger.log(f"Intermediate Target Step {total_step_number+1} -Action {action} -Finished {finished} -gen {config.TARGET_NORMAL_GENERATION[total_step_number+1]} -prompt {config.TARGET_NORMAL_PROMPT[total_step_number+1]}")
    target_logger.log(f"Target Step {total_step_number+1}: -time {t_time}")
    config.TARGET_NORMAL_TIME[total_step_number+1] = t_time

    if not mismatch_state.mismatch_detected.is_set() and ta[0] >= k-1:
        mismatch_state.mismatch_step_id = ta[0]
        mismatch_state.mismatch_detected.set()

    return tas, printed_ids


async def onebreakingpoint_speculative_planning(
    args, mismatch_state, executor, encoding, app_assistant, tar_assistant, previous_steps, current_step, approximation_logger, target_logger, logger, train_logger, collector
):
    sas = []
    tas = []
    target_tasks = []
    printed_ids = []
    mismatch_state.mismatch_detected.clear()
    pred_k = 1 if config.ENABLE_PRED else args.k
    first = True
    
    i = 0
    prediction_task = None
    while i < pred_k:
        if mismatch_state.mismatch_detected.is_set():
            break
        if config.ENABLE_PRED and first:
            prediction_task = asyncio.create_task(executor.async_predict(approximation_logger, args.offset))
            config.PENDING_BACKGROUND_TASKS.append(prediction_task)
        break_out_approximation = False

        tas.append([])
        printed_ids.append([])

        a_start_time = time.time()
        approximation = asyncio.create_task(
            A_generate(app_assistant, current_step + i, approximation_logger=approximation_logger, collector=collector, start=a_start_time),
            name=f"approximation_{current_step+i}",
        )
        if mismatch_state.mismatch_detected.is_set():
            break
        # return thought, action, finished
        t_start_time = time.time()
        target = asyncio.create_task(
            T_generate(
                tar_assistant,
                prediction_task,
                current_step + i,
                tas,
                sas,
                previous_steps,
                target_tasks=target_tasks,
                printed_ids=printed_ids,
                current_step=current_step,
                logger=logger,
                target_logger=target_logger,
                collector=collector,
                mismatch_state=mismatch_state,
                start = t_start_time
            )
        )
        target_tasks.append(target)

        concurrent_api_calls = concurrent_calls()
        if concurrent_api_calls >= config.MAX_CONCURRENT_CALLS:
            config.MAX_CONCURRENT_CALLS = concurrent_api_calls

        try:
            action, observation = await approximation
            sa = [action, observation]
            sas.append(sa)
            if mismatch_state.mismatch_detected.is_set():
                break
        except asyncio.CancelledError as e:
            pass

        # if sa == terminate, and ta == terminate, we break the loop
        # if sa == terminate, and ta != terminate, we also break the loop
        # thus as long as sa == terminate, we break the loop
        if sa[1] is True or sa[0].lower == 'terminate': ## terminate
            break_out_approximation = True

        # tas is now a list of lists, so we need to flatten it in order to compare with sas
        flatten_tas = []
        for t in tas:
            if t:
                flatten_tas += t
        flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
        # halt the ongoing approximation loop
        for t in flatten_tas:
            if len(sas) > t[0]:
                # if approximation action != target action
                if not judge_to_be_true(sas[t[0]][0], t[1][0]) or t[1][0].lower() == "terminate" or t[1][1]==True:
                    break_out_approximation = True
                    if not mismatch_state.mismatch_detected.is_set():
                        mismatch_state.mismatch_step_id = t[0]
                        mismatch_state.mismatch_detected.set()
                    for process_id, one_task in enumerate(target_tasks):
                        if (
                            not one_task.cancelled()
                            and not one_task.done()
                            and process_id > t[0]
                        ):
                            target_logger.log(f"Cancel Task {len(previous_steps)+process_id+1}")
                            await cancel(one_task)
                    break

        if break_out_approximation:
            break

        if config.ENABLE_PRED and first:
            pred_k = await prediction_task
            config.PREDICT_K.append(pred_k)
            config.PREDICT_TOTAL += 1
            pred_k = max(pred_k, 0)
            first = False
        i += 1

    # after halting the approximation loop
    # we need to collect the target results
    # organize to sas, see how much we want to preserve
    # SHOULD NOT exclude finished tasks, because exceptions are only thrown when tasks are finished
    await mismatch_state.mismatch_detected.wait()
    target_logger.log(f"Breaking point stops at step {len(previous_steps)+1+mismatch_state.mismatch_step_id}. Start Cancellation.")
    for process_id, one_task in enumerate(target_tasks):
        if not one_task.cancelled() and not one_task.done() and process_id > mismatch_state.mismatch_step_id:
            # cancel ongoing target task after mismatch
            target_logger.log(f'Cancel task {len(previous_steps)+process_id+1}')
            await cancel(one_task)

    pending_tasks = [t for t in target_tasks if not t.cancelled()]

    while pending_tasks:
        break_while_loop = False
        try:
            if [pending_task.done() for pending_task in pending_tasks] == [True] * len(
                pending_tasks
            ):
                break_while_loop = True
                try:
                    await asyncio.gather(*pending_tasks, return_exceptions=False)
                    break
                except:
                    break
            # should not await cancelled tasks
            # return_exceptions=False is also the default value
            await asyncio.gather(*pending_tasks, return_exceptions=False)
            break_while_loop = True
            break
        except Exception as e:
            if str(e) == "terminate the whole process!":
                # cancel all pending tasks because we have already got TERMINATE
                for process_id, one_task in enumerate(pending_tasks):
                    if not one_task.cancelled() and not one_task.done() and process_id > mismatch_state.mismatch_step_id:
                        target_logger.log(f"cancel task at line 552 {len(previous_steps)+process_id+1}")
                        await cancel(one_task)
                # organize the results and return the final results
                flatten_tas = []
                for t in tas:
                    if t:
                        flatten_tas += t
                flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
                printed_ids, tas = simulate_leftover_interaction(
        collector, sas, tas, flatten_tas, printed_ids, logger, tar_assistant, previous_steps
    )

                # get the final tas result
                flatten_tas = []
                for t in tas:
                    if t:
                        flatten_tas += t
                flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
                for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
                    if t[0] == step_number and not judge_to_be_true(s[0], t[1][0]):
                        tar_assistant.execute(t[1][0])
                        to_replace_action = [
                            flatten_tas[step_number][1][0],
                            tar_assistant.current_observation,
                        ]
                        sas = sas[:step_number] + [to_replace_action]
                        app_assistant.update_scratchpad(
                            sas[-1][0], sas[-1][1], len(previous_steps) + step_number
                        )
                        config.PREDICT_CORRECT += collector.build_trajectory(target_logger, config.PREDICT_K)
                        config.PREDICT_K = [] 
                        if config.ENABLE_TRAIN:
                            if config.BUILD_TRAJ_TIMES == 0:
                                train_task = asyncio.create_task(executor.async_train(train_logger))
                                config.PENDING_BACKGROUND_TASKS.append(train_task)
                            config.BUILD_TRAJ_TIMES = (config.BUILD_TRAJ_TIMES + 1) % config.TRAIN_INTERVAL
                        break   
                return sas
            else:
                # cancel t_j for j > i if t_i != s_i
                if [pending_task.done() for pending_task in pending_tasks] == [
                    True
                ] * len(pending_tasks):
                    break_while_loop = True
                    try:
                        await asyncio.gather(*pending_tasks, return_exceptions=False)
                        break
                    except:
                        break
                if break_while_loop:
                    break
                mistaken_process_id = int(str(e)[-1])
                
                for process_id, one_task in enumerate(pending_tasks):
                    if (
                        not one_task.cancelled()
                        and not one_task.done()
                        and process_id > mistaken_process_id
                    ):
                        target_logger.log(f"cancel task at line 608 {len(previous_steps)+process_id+1}")
                        await cancel(one_task)

                pending_tasks = [
                    t for process_id, t in enumerate(target_tasks)
                    if not t.cancelled() and process_id != mistaken_process_id
                ]

        if break_while_loop:
            break

    
    # get user input or interruption
    flatten_tas = []
    for t in tas:
        if t:
            flatten_tas += t
    flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
    printed_ids, tas = simulate_leftover_interaction(
        collector, sas, tas, flatten_tas, printed_ids, logger, tar_assistant, previous_steps
    )

    # get the final tas result
    flatten_tas = []
    for t in tas:
        if t:
            flatten_tas += t
    flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
    
 
    all_match = True
    for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
        if (len(previous_steps)+step_number+1 >= config.MAX_STEP) or (t[0] == step_number and (not judge_to_be_true(s[0], t[1][0]) or (t[1][0].lower() == "terminate" or t[1][1]==True))):  # t[1] != s:
            all_match = False
            config.PREDICT_CORRECT += collector.build_trajectory(target_logger, config.PREDICT_K)
            config.PREDICT_K = [] 
            
            if config.ENABLE_TRAIN:
                if config.BUILD_TRAJ_TIMES == 0:
                    train_task = asyncio.create_task(executor.async_train(train_logger))
                    config.PENDING_BACKGROUND_TASKS.append(train_task)
                config.BUILD_TRAJ_TIMES = (config.BUILD_TRAJ_TIMES + 1) % config.TRAIN_INTERVAL
            
            t1 = time.time()
            tar_assistant.execute(t[1][0])
            t2 = time.time()
            config.TARGET_NORMAL_TIME[len(previous_steps)+step_number+1] += round(t2-t1, 2)
            to_replace_action = [
                flatten_tas[step_number][1][0],
                tar_assistant.current_observation,
            ]
            sas = sas[:step_number] + [to_replace_action]
            app_assistant.update_scratchpad(
                sas[-1][0], sas[-1][1], len(previous_steps) + step_number
            )

            break
    
    if all_match:
        sas = sas = sas[:len(flatten_tas)]
    return sas


async def speculative_planning(args, executor, encoding, app_assistant, tar_assistant, approximation_logger, target_logger, logger):
    train_logger = None
    if config.ENABLE_TRAIN:
        pred_type = "dyn_k" if args.pred else "fix_k"
        log_dir = f"../data/travel_planner/{args.approx_type}_{args.target_type}/{args.model_type}/{pred_type}"
        train_logger = Logger(f'{log_dir}/tau_{args.tau}_offset_{args.offset}/train_datapoint{task_id}.log', on=True)
    
    begin_time = datetime.now()
    collector = executor.collector
    mismatch_state = SharedState()
    await mismatch_state.initialize()
    steps = []
    breaking_points = 0
    i = 0

    while True:
        result = await onebreakingpoint_speculative_planning(
            args, mismatch_state, executor, encoding, app_assistant, tar_assistant, steps, len(steps), approximation_logger, target_logger, logger, train_logger, collector
        )

        steps += result
        breaking_points += 1
        i += len(result)

        # if the last action is terminate, we break the generation process
        if result[-1][0].lower() == "terminate" or result[-1][1] is True or len(steps) >= config.MAX_STEP:
            break

    end_time = datetime.now()
    logger.log(f"{end_time} - {begin_time} = {end_time - begin_time}")
    config.TOTAL_SP_TIME = round((end_time - begin_time).total_seconds(), 2)
    # wait for training process end
    if config.PENDING_BACKGROUND_TASKS:
        await asyncio.gather(*config.PENDING_BACKGROUND_TASKS)
    return steps

def run_one_task(args, task_id, numbers, query_data_list, executor, encoding, approximation_agent, target_agent, traj_file):
    task_id = numbers[task_id] - 1
    # select query
    query = query_data_list[task_id]["query"]
    executor.set_initial_task_prompt(query)
    print(f"Query is {query}")
    target_agent.query = query
    approximation_agent.query = query
    
    config.MAX_CONCURRENT_CALLS = 0
    config.TOTAL_APPROXIMATION_CALLS = 0
    config.TOTAL_CORRECT_APPROXIMATION_CALLS = 0

    config.TOTAL_TOKEN_GENERATION = 0
    config.TOTAL_TOKEN_PROMPT = 0
    config.USERINPUT=False

    config.TARGET_NORMAL_PROMPT = defaultdict(int)
    config.TARGET_NORMAL_GENERATION = defaultdict(int)
    config.TARGET_SP_PROMPT = 0
    config.TARGET_SP_GENERATION = 0

    config.APPROX_SP_PROMPT = 0
    config.APPROX_SP_GENERATION = 0
    config.APPROX_NORMAL_PROMPT = defaultdict(int)
    config.APPROX_NORMAL_GENERATION = defaultdict(int)

    config.TOTAL_SP_TIME = 0
    config.TARGET_NORMAL_TIME = {}
    config.APPROX_NROMAL_TIME = {}

    config.PREDICT_K = []
    config.PREDICT_CORRECT = 0
    config.PREDICT_TOTAL = 0
    config.PENDING_BACKGROUND_TASKS = []

    random.seed(2)
    pred_type = "dyn_k" if args.pred else "fix_k"
    log_dir = f"../data/travel_planner/{args.approx_type}_{args.target_type}/{args.model_type}/{pred_type}"
    if config.ENABLE_TRAIN: 
        logger = Logger(f'{log_dir}/tau_{args.tau}_offset_{args.offset}/simulation_datapoint{task_id}.log', on=True)
        target_logger = Logger(f'{log_dir}/tau_{args.tau}_offset_{args.offset}/target_datapoint{task_id}.log', on=True)
        approximation_logger = Logger(f'{log_dir}/tau_{args.tau}_offset_{args.offset}/approximation_datapoint{task_id}.log', on=True)
    else:
        logger = Logger(f'{log_dir}/k_{args.k}/simulation_datapoint{task_id}.log', on=True)
        target_logger = Logger(f'{log_dir}/k_{args.k}/target_datapoint{task_id}.log', on=True)
        approximation_logger = Logger(f'{log_dir}/k_{args.k}/approximation_datapoint{task_id}.log', on=True)
    logger.log(f'task description: {query}')

    

    # run the speculative planning
    steps = asyncio.run(
        speculative_planning(args, executor, encoding, approximation_agent, target_agent, approximation_logger, target_logger, logger)
    )    

    # record the metrics
    logger.log("final result for the speculative planning " + str([s[0] for s in steps]))
    logger.log(
        "max concurrent calls: " + str(config.MAX_CONCURRENT_CALLS - 1)
    )  # speculative_planning will add one more call
    
    sp_plan_token = config.TOTAL_TOKEN_GENERATION
    step_num = len(steps)

    normal_plan_time       = sum(config.TARGET_NORMAL_TIME[i] for i in range(1, step_num+1))
    normal_app_time        = sum(config.APPROX_NROMAL_TIME[i] for i in range(1, step_num+1))
    normal_tar_generation  = sum(config.TARGET_NORMAL_GENERATION[i] for i in range(1, step_num+1))
    normal_app_generation  = sum(config.APPROX_NORMAL_GENERATION[i] for i in range(1, step_num+1))
    normal_plan_generation = normal_tar_generation + normal_app_generation
    normal_tar_prompt      = sum(config.TARGET_NORMAL_PROMPT[i] for i in range(1, step_num+1))
    normal_app_prompt      = sum(config.APPROX_NORMAL_PROMPT[i] for i in range(1, step_num+1))
    normal_plan_prompt     = normal_app_prompt + normal_tar_prompt
    
    logger.log('normal approx token prompt: ' + str(normal_app_prompt))
    logger.log('normal approx token generation: ' + str(normal_app_generation))

    logger.log('sp approx token prompt: ' + str(config.APPROX_SP_PROMPT))
    logger.log('sp approx token generation: ' + str(config.APPROX_SP_GENERATION))

    logger.log('normal target token prompt: ' + str(normal_tar_prompt))
    logger.log('normal target token generation: ' + str(normal_tar_generation))

    logger.log('sp target token prompt: ' + str(config.TOTAL_TOKEN_PROMPT - config.APPROX_SP_PROMPT))
    logger.log('sp target token generation: ' + str(sp_plan_token - config.APPROX_SP_GENERATION))

    logger.log('total sp token prompt: ' + str(config.TOTAL_TOKEN_PROMPT))
    logger.log('total sp token generation: ' + str(sp_plan_token))

    logger.log('normal target step time: ' + str(normal_plan_time))
    logger.log(
        "accuracy of approximation agent: "
        + str(
            config.TOTAL_CORRECT_APPROXIMATION_CALLS / config.TOTAL_APPROXIMATION_CALLS
        )
    )
    avg_sp_token = round(sp_plan_token/step_num, 2)
    logger.log('sp token generation/step: ' + str(avg_sp_token))
    avg_normal_token = round(normal_plan_generation/step_num, 2)
    logger.log('normal token generation/step: ' + str(avg_normal_token))
    logger.log('generation token cost increased: ' + str(round((avg_sp_token/avg_normal_token-1)*100, 2))+"%.")
    
    avg_sp_prompt = config.TOTAL_TOKEN_PROMPT / step_num
    logger.log('sp prompt/step: ' + str(round(avg_sp_prompt, 2)))
    avg_normal_prompt = round(normal_plan_prompt/step_num, 2)
    logger.log('normal prompt/step: ' + str(avg_normal_prompt))
    logger.log('prompt token cost increased: ' + str(round((avg_sp_prompt/avg_normal_prompt-1)*100, 2))+"%.")

    avg_sp_time = round(config.TOTAL_SP_TIME/step_num, 2)
    logger.log('sp time/step: ' + str(avg_sp_time))
    avg_normal_time = round(normal_plan_time/step_num, 2)
    avg_approx_time = round(normal_app_time/step_num, 2)
    logger.log('normal target time/step: ' + str(avg_normal_time))
    logger.log('normal approx time/step: ' + str(avg_approx_time))
    logger.log('time decreased by: ' + str(round((1-avg_sp_time/avg_normal_time)*100, 2))+"%.")
    if args.pred:
        logger.log(f'predictor acc: {round(config.PREDICT_CORRECT / config.PREDICT_TOTAL, 2)}')
    logger.log(f'step number: {step_num}')

    if not args.pred:
        logger.log(f'k = {args.k}')
        traj_file = traj_file.format(task_id)
    else: logger.log('dynamic k')
    executor.save_trajectory(traj_file, config.ENABLE_TRAIN)



if __name__ == "__main__":
    encoding = tiktoken.get_encoding("cl100k_base")

    os.environ['DEEPSEEK_API_KEY'] = ""
    os.environ['OPENAI_API_KEY'] = ""
    os.environ['GOOGLE_API_KEY'] = ""

        
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument('--approx_type', type=str, default='direct', help='cot, direct')
    parser.add_argument('--target_type', type=str, default='react', help='react, multi_agent')
    parser.add_argument("--model_type", type=str, default="gpt-4.1-mini", help='deepseek-chat, gpt-4.1-mini')
    parser.add_argument('--pred', action='store_true', help='enable speculative planning with predictor')
    parser.add_argument('--no-pred', dest='pred', action='store_false', help='disable speculative planning with predictor')
    parser.set_defaults(pred=True)
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="number of approximation steps to generate everytime",
    )
    parser.add_argument('--tau', type=float, default=0.5, help='tau for expectile loss')
    parser.add_argument('--lr', type=float, default=1e-5, help='online learning lr')
    parser.add_argument('--ep', type=int, default=3, help='online learning epoch per train')
    parser.add_argument('--bf', type=int, default=2500, help='online learning buffer size')
    parser.add_argument('--bs', type=int, default=16, help='online learning batch size')
    parser.add_argument('--gma', type=float, default=1, help='online learning gamma for lambda return calculation')
    parser.add_argument('--lmd', type=float, default=0.95, help='online learning lambda for lambda return calculation')
    parser.add_argument('--load', dest='load', action='store_true', help='load previous trajectory and model')
    parser.add_argument('--no-load', dest='load', action='store_false', help='do not load previous trajectory and model')
    parser.add_argument('--freq', type=int, default=1, help='online learning frequency')
    parser.add_argument('--s_task', type=int, default=1, help='online learning start task id')
    parser.add_argument('--offset', type=int, default=0, help='inference biased offset for k')
    
    parser.set_defaults(load=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    config.TRAIN_INTERVAL = args.freq
    config.BUILD_TRAJ_TIMES = 0
    config.MAX_STEP = 8

    if args.set_type == "validation":
        query_data_list = load_dataset("osunlp/TravelPlanner", "validation")[
            "validation"
        ]
    elif args.set_type == "test":
        query_data_list = load_dataset("osunlp/TravelPlanner", "test")["test"]

    numbers = [i for i in range(1, len(query_data_list) + 1)]
    
    tools_list = [
        "notebook",
        "flights",
        "attractions",
        "accommodations",
        "restaurants",
        "googleDistanceMatrix",
        "planner",
        "cities",
    ]

    app_model_type = args.model_type
    tar_model_type = args.model_type if args.model_type == "gpt-4.1-mini" else "deepseek-reasoner"
    # setup target agent
    if args.target_type == "react":
        target_agent = ReactAgent(
            None,
            tools=tools_list,
            max_steps=config.MAX_STEP,
            react_llm_name=tar_model_type,
            planner_llm_name=tar_model_type,
        )
    elif args.target_type == "multi_agent":
        target_agent = MultiAgent(
            None,
            tools=tools_list,
            max_steps=config.MAX_STEP,
            react_llm_name=tar_model_type,
            planner_llm_name=tar_model_type,
        )
    else: target_agent = None
    config.TARGET_TYPE = args.target_type
    
    # setup approximation agent
    if args.approx_type == "direct":
        approximation_agent = DirectAgent(
            None,
            tools=tools_list,
            max_steps=config.MAX_STEP,
            react_llm_name=app_model_type,
            planner_llm_name=app_model_type,
        )
    elif args.approx_type == "cot":
        approximation_agent = CoTAgent(
            None,
            tools=tools_list,
            max_steps=config.MAX_STEP,
            react_llm_name=app_model_type,
            planner_llm_name=app_model_type,
        )
    else: approximation_agent = None
    # online learning preparations
    model_path = "distilbert-base-uncased"
    bert_model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DistilBERTValueFunction(bert_model).to(device)
    
    if args.pred: # dyn k
        traj_dir = f"../trajectory/travel_planner/online_traj/{args.approx_type}_{args.target_type}/{args.model_type}"
        os.makedirs(traj_dir, exist_ok=True)
        traj_file = f"{traj_dir}/tau_{args.tau}_offset_{args.offset}.ndjson"
    else: # fix k
        traj_dir = f"../trajectory/travel_planner/{args.approx_type}_{args.target_type}/{args.model_type}/fix_k_{args.k}"
        os.makedirs(traj_dir, exist_ok=True)
        traj_file = f"{traj_dir}/task_{{}}.json"
    
    ckpt_dir = f"../ckpt/travel_planner/online/{args.approx_type}_{args.target_type}/{args.model_type}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/tau_{args.tau}_offset_{args.offset}.pth"
    executor = OnlineLearningExecutor(
        device=device,
        model_save_path=ckpt_path,
        model=model,
        tokenizer=tokenizer,
        buffer_size=args.bf,
        batch_size_steps=args.bs,
        lambda_=args.lmd,
        gamma=args.gma,
        lr=args.lr,
        epoch_per_train=args.ep,
        load=args.load,
        traj_file=traj_file,
        tau = args.tau
    )
    

    # run the speculative planning for multiple tasks
    task_ids = list(range(args.s_task, 180))

    config.ENABLE_TRAIN = args.pred
    config.ENABLE_PRED = args.pred
    config.WARMUP = 0 # TASK number for online warmup
    
    warmup_task = 0
    for task_id in task_ids:
        if warmup_task >= config.WARMUP:
            config.ENABLE_PRED = args.pred
        run_one_task(args, task_id, numbers, query_data_list, executor, encoding, approximation_agent, target_agent, traj_file)
        warmup_task += 1
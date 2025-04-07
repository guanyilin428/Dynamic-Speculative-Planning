###################################################
# This version is currently the most optimzed one
###################################################

import asyncio
import time
from datetime import datetime
import random
import subprocess
import random
import json 
import torch
import autogen 
import argparse 
from color import slow_type_approximation, slow_type_target

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, GroupChat, GroupChatManager
from autogen.agentchat.contrib.agent_builder import AgentBuilder

from utils import Logger, cancel, register_async_handler
import config
import os 


import openai

import tiktoken
from tiktoken.load import load_tiktoken_bpe
# encoding = tiktoken.get_encoding("cl100k_base")
from openai import OpenAI

bpe_path = "cl100k_base.tiktoken"
bpe_ranks = load_tiktoken_bpe(bpe_path)
# os.environ['OPENAI_API_KEY'] = "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"
# client = OpenAI(api_key="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")
# print(os.environ['OPENAI_API_KEY'])
# openai.api_key = os.getenv("OPENAI_API_KEY")

# asyncio.get_event_loop().set_debug(True)

from transformers import AutoModel, AutoTokenizer
from OpenAGI import DistilBERTValueFunction, OnlineTrajectoryCollector, OnlineLearningExecutor, SharedState


def concurrent_calls():
    tasks = asyncio.all_tasks()
    pending_tasks = [t for t in tasks if not t.done() and not t.cancelled()]
    return len(pending_tasks)

### functions that can be customized
def parse_user_input(user_input, s, t):
    if user_input == 'approximation answer':
        return s
    elif user_input == 'target answer':
        return t[0][1]
    elif user_input == '':
        return t[0][1]
    else:
        return user_input

### functions that can be customized
def interaction_function(collector, sas, tas, to_print_id, logger, target_logger, prev_steps, target_tasks):
    s = sas[to_print_id]
    t = tas[to_print_id]
    target_logger.log(f'Target: Step {t[0][0] + len(prev_steps)+1} - {t[0][1]}')
    
    # Add to online trajectory collector
    cur_time = time.time()
    timestamp = datetime.fromtimestamp(cur_time)
    source = "Target"
    step = t[0][0] + len(prev_steps) + 1
    desc = t[0][1]
    collector.record_step(timestamp, source, step, desc)
    # print(f'{timestamp} Target: Step {t[0][0] + len(prev_steps)+1} - {t[0][1]}')
    
    try:
        config.TOTAL_APPROXIMATION_CALLS += 1
        if judge_to_be_true(s, t[0][1]):
            config.TOTAL_CORRECT_APPROXIMATION_CALLS += 1
            logger.log(f'The target agent thinks step {len(prev_steps) + to_print_id+1} should be '+ str(t[0][1]) + ', which agrees with the approximation agent.')
            try:
                logger.log(f'The approximation agent thinks step {len(prev_steps) + to_print_id+2} should be ' + sas[to_print_id+1])
                config.HIL_INTERACTION = to_print_id+1
                register_async_handler(target_tasks=target_tasks)
            except:
                pass
        else:
            logger.log(f'The target agent thinks step {len(prev_steps) + to_print_id+1} should be '+ str(t[0][1]) + ', correcting what the approximation agent thinks which is ' + str(s) + '.')
        user_input = ''
        user_input = parse_user_input(user_input, s, t)
    except KeyboardInterrupt as e:
        user_input = input('')

        user_input = parse_user_input(user_input, s, t)
        if not judge_to_be_true(user_input, str(t[0][1])):
            logger.log(f'Since you think the action should be {user_input} ... we will follow your suggestion :)')
        logger.log('-------------------')

    return user_input

### functions that can be customized
def judge_to_be_true(s, t):
    if s == t:
        return True
    else:
        return False

############# autogen code for speculative planning #############
def load_data(args):
    with open("data/openagi_task_description.txt", "r") as f:
        data = f.read()
    data = [t.strip() for t in data.split("\n")]
    return data

def parse_response(response):
    if '<' in response and '>' in response and '</' in response:
        # find the last tag
        all_starts = [i for i in range(len(response)-1) if response[i] == '<' and response[i+1] != '/']
        all_ends = [i for i in range(len(response)-1) if response[i:i+2] == '</']
        start = all_starts[-1]
        end = all_ends[-1]
        return response[start:end].replace('<tool>', '').replace('<', '').replace('>', '')
    elif '**' in response and response.count('**') >= 2:
        start = response.index('**') + len('**')
        response = response[start:]
        end = response.index('**')
        return response[:end].replace('<', '').replace('>', '')
    else:
        return ''

def ordinal(n):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

def simulate_within_T_interaction(collector, sas, tas, flatten_tas, printed_ids, logger, target_logger, prev_steps, target_tasks):
    # conduct on-time printing out
    tas_ids = [t[0] for t in flatten_tas]
    flatten_printed_ids = []
    for ids in printed_ids:
        if ids:
            flatten_printed_ids += ids
    tas_length = len(tas_ids)
    for l in range(0, tas_length+1):
        if not(tas_ids[:l] == list(range(len(flatten_tas)))[:l] and len(sas) >= len(flatten_tas[:l])):
            tas_length = l-1
            break

    # if the printed ids contain wrong results
    contain_wrong_result = False
    if tas_length > 0:
        for printed_id in flatten_printed_ids:
            if not judge_to_be_true(sas[printed_id], tas[printed_id][0][1]):
                contain_wrong_result = True

    if tas_length > 0 and not contain_wrong_result:
        tas_ids = tas_ids[:tas_length]
        to_print_ids = list(set(tas_ids) - set(flatten_printed_ids))
        for order_id, to_print_id in enumerate(to_print_ids):
            if order_id > 0:
                if not judge_to_be_true(sas[to_print_ids[order_id-1]], tas[to_print_ids[order_id-1]][0][1]):
                    break
            printed_ids[to_print_id].append(to_print_id)
            user_input = interaction_function(collector, sas, tas, to_print_id, logger, target_logger, prev_steps, target_tasks)
            # print('user_input', user_input.lower())
            if str(user_input) == str(tas[to_print_id][0][1]) and user_input.lower() != 'terminate':
                continue 
            elif str(user_input) == str(tas[to_print_id][0][1]) and user_input.lower() == 'terminate':
                return printed_ids, tas
            else:
                change_one_tas_position = list(tas[to_print_id][0])
                change_one_tas_position[1] = user_input
                tas[to_print_id][0] = change_one_tas_position
                return printed_ids, tas

    ## finish printing out
    return printed_ids, tas

def simulate_leftover_interaction(collector, sas, tas, flatten_tas, printed_ids, logger, target_logger, prev_steps, target_tasks):
    # print('simulate_leftover_interaction')
    # when sas is slower than tas 
    # we need to print out the extra here
    flatten_printed_ids = []
    for ids in printed_ids:
        if ids:
            flatten_printed_ids += ids
    print_leftover = True
    if len(sas) > len(flatten_printed_ids) and len(flatten_tas) > len(flatten_printed_ids):
        for printed_id in flatten_printed_ids:
            if not judge_to_be_true(sas[printed_id], tas[printed_id][0][1]):
                print_leftover = False
    else:
        print_leftover = False

    # if the printed ids contain wrong results
    contain_wrong_result = False
    for printed_id in flatten_printed_ids:
        if not judge_to_be_true(sas[printed_id], tas[printed_id][0][1]):
            contain_wrong_result = True

    if print_leftover and not contain_wrong_result:
        for print_id in range(len(flatten_printed_ids), min(len(sas),len(tas))):
            if tas[print_id]:
                user_input = interaction_function(collector, sas, tas, print_id, logger, target_logger, prev_steps, target_tasks)
                printed_ids[print_id].append(print_id)
                if str(user_input) == str(tas[print_id][0][1]):
                    continue 
                else:
                    change_one_tas_position = list(tas[print_id][0])
                    change_one_tas_position[1] = user_input
                    tas[print_id][0] = change_one_tas_position
            else:
                break
            if not judge_to_be_true(sas[print_id], tas[print_id][0][1]):
                break
            if user_input.lower() == 'terminate':
                break

    return printed_ids, tas

async def A_generate(collector, encoding, assistant, prompt, total_step_number, logger, approximation_logger):
    start = time.time()
    prompt += f"\n\nDirectly tell me what **the ONE NEXT action step** based on the current action trajectory should be. (Remember to use xml tag <tool> and </tool> for formatting.)\nWhat should be the action in Step {total_step_number+1}?\nTarget won't terminate easily so don't choose <tool>TERMINATE</tool> unless you are very confident. \nStep {total_step_number+1}:"
    n = 0
    # approximation_logger.log(f'Approximation prompt: Step {total_step_number+1} - {prompt}')

    # approximation_logger.log(prompt)
    while True:
        try:
            n += 1
            if n >= 10:
                result = ''
                return result
            prompt_token = len(encoding.encode(prompt))
            config.TOTAL_TOKEN_PROMPT += prompt_token
            config.APPROX_TOTAL_PROMPT += prompt_token
            approximation_logger.log(f'Approximation: Step {total_step_number+1} -prompt token {prompt_token}')
            response = await assistant.a_generate_reply(messages=[{'content':prompt, 'role':'user'}])
            
            # approximation_logger.log(f"Approximation tokens: {len(encoding.encode(response))}") 
            app_tokens = len(encoding.encode(response))
            config.TOTAL_TOKEN_GENERATION += app_tokens
            config.APPROX_TOTAL_GENERATION += app_tokens
            
            result = parse_response(response)
            end = time.time()
            
            # add approximation task to online trajectory collector
            timestamp = datetime.fromtimestamp(end)
            source = "Approximation"
            step = total_step_number+1
            desc = result
            collector.record_step(timestamp, source, step, desc)
            
            a_time = round(end-start, 2)
            approximation_logger.log(f'Approximation: Step {total_step_number+1} - {result} -time {str(a_time)} -token {app_tokens}')
            # print(f'{timestamp} Approximation: Step {total_step_number+1} - {result}')
            # approximation_logger.log(f'approximation: step time for {step}: '+ str(end-start))
            
            return result
        except:
            continue

async def ReAct(encoding, assistant, prompt, total_step_number, target_logger):
    prompt += f"\n\nCarefully think about **the ONE NEXT action step** based on the current action trajectory."
    prompt += f"\nGenerate thought only.\nThought {total_step_number}:"

    target_logger.log('react launch api for thought.')

    prompt_token = len(encoding.encode(prompt))
    config.TARGET_TOTAL_PROMPT[total_step_number+1] = prompt_token
    config.TOTAL_TOKEN_PROMPT += prompt_token
    target_logger.log(f"Target step {total_step_number+1} thought prompt token: {prompt_token}")
    thought = await assistant.a_generate_reply(messages=[{'content':prompt, 'role':'user'}])
    thought_token = len(encoding.encode(thought))
    config.TOTAL_TOKEN_GENERATION += thought_token
    config.TARGET_TOTAL_GENERATION[total_step_number+1] = thought_token
    target_logger.log(f"Target step {total_step_number+1} thought generation token: {thought_token}")

    prompt += " " + thought
    prompt += f"\nGenerate Action only based on thoughts. Remember to use xml tag <tool> and </tool> for formatting. \nAction {total_step_number}:"
    # generate action based on thought
    target_logger.log('react launch api for response.')
    prompt_token = len(encoding.encode(prompt))
    config.TARGET_TOTAL_PROMPT[total_step_number+1] += prompt_token
    config.TOTAL_TOKEN_PROMPT += prompt_token
    target_logger.log(f"Target step {total_step_number+1} response prompt token: {prompt_token}")
    response = await assistant.a_generate_reply(messages=[{'content':prompt, 'role':'user'}])
    response_token = len(encoding.encode(response))
    config.TOTAL_TOKEN_GENERATION += response_token
    config.TARGET_TOTAL_GENERATION[total_step_number+1] += response_token
    target_logger.log(f"Target step {total_step_number+1} response generation token: {response_token}")

    return response

async def postprocess_T_generation(prediction_task, mismatch_state, collector, result, total_step_number, tas, sas, target_tasks, printed_ids=[[]], current_step=0, logger=None, target_logger=None, prev_steps=[]): 
    in_step_number = total_step_number - current_step
    tas[in_step_number].append((in_step_number,result))

    # cancel next target_tasks after which we know is incorrect
    # but there maybe unknown result before this point, so we don't kill all processes
    flatten_tas = []
    for t in tas:
        if t:
            flatten_tas += t
    flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
    printed_ids, tas = simulate_within_T_interaction(collector, sas, tas, flatten_tas, printed_ids, logger, target_logger, prev_steps, target_tasks)

    # if the target result is terminate, we break the loop
    flatten_ids = [t[0] for t in flatten_tas]
    if flatten_ids == list(range(len(flatten_ids))):
        for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
            if t[0] == step_number and t[1].lower() == 'terminate':
                end = time.time()
                mismatch_state.mismatch_detected.set()
                mismatch_state.mismatch_step_id = ta[0]
                raise Exception('terminate the whole process!')

    # if it is a wrong result
    # we break out the target processes and cancel processes that comes after it
    flatten_tas = []
    for t in tas:
        if t:
            flatten_tas += t
    flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
    for ta in flatten_tas:
        if len(sas) > ta[0]:
            if not sas[ta[0]] == ta[1]:
                # target_logger.log(f"task {len(prev_steps)+ta[0]+1} mismatched.")
                if not mismatch_state.mismatch_detected.is_set():
                    mismatch_state.mismatch_step_id = ta[0]
                    mismatch_state.mismatch_detected.set() # mismatch occurs
                end = time.time()
                pending_approximation_tasks = [t for t in asyncio.all_tasks() if not t.cancelled() and not t.done() and t not in target_tasks and t.get_name().startswith('approximation')]
                for pending_approximation_task in pending_approximation_tasks:
                    # mismatch occurs, cancel ongoing approximation tasks
                    await cancel(pending_approximation_task)
                raise Exception(f'approximation error happen in step {total_step_number} for current step {current_step}, the target id is {ta[0]}')
    if args.pred:
        k = await prediction_task
    else: k = args.k
    if not mismatch_state.mismatch_detected.is_set() and ta[0] >= k-1:
        # all match in current breakingpoint
        mismatch_state.mismatch_detected.set()
        mismatch_state.mismatch_step_id = ta[0]

    return tas, printed_ids

async def T_generate(args, prediction_task, mismatch_state, collector, encoding, assistant, prompt, total_step_number, tas, sas, target_tasks, printed_ids, current_step, logger, target_logger, prev_steps):
    start = time.time()
    result = None
    if args.target_type == 'direct':
        prompt += f"\n\nDirectly tell me what **the ONE NEXT action step** based on the current action trajectory should be. (Remember to use xml tag <tool> and </tool> for formatting.)\nWhat should be the action in Step {total_step_number+1}?\n\nStep {total_step_number+1}:"
    else:
        prompt += f"\n\nCarefully think about **the ONE NEXT action step** based on the current action trajectory, by first providing a clear reasoning chain. And then decide which tool to use for the current step. (Remember to use xml tag <tool> and </tool> for formatting.)\nWhat should be the action in Step {total_step_number+1}?\n\nStep {total_step_number+1}:"
    
    try:
        # call agent to generate the response
        n = 0
        while True:
            try:
                n += 1
                if n >= 10:
                    result = ''
                    break
                if args.target_type == 'react':
                    response = await ReAct(encoding, assistant, prompt, total_step_number, target_logger)
                else:
                    # if mismatch_state.mismatch_detected.is_set() and mismatch_state.mismatch_step_id < 
                    # log prompt token
                    prompt_token = len(encoding.encode(prompt))
                    config.TARGET_TOTAL_PROMPT[total_step_number+1] = prompt_token
                    config.TOTAL_TOKEN_PROMPT += prompt_token
                    response = await assistant.a_generate_reply(messages=[{'content':prompt, 'role':'user'}])
                    response_token = len(encoding.encode(response))
                    config.TOTAL_TOKEN_GENERATION += response_token
                    config.TARGET_TOTAL_GENERATION[total_step_number+1] = response_token
                result = parse_response(response)
                break
            except:
                await asyncio.sleep(0.1)
                continue
        tas, printed_ids = await postprocess_T_generation(prediction_task, mismatch_state, collector, result, total_step_number, tas, sas, target_tasks, printed_ids=printed_ids, current_step=current_step, logger=logger, target_logger=target_logger, prev_steps=prev_steps)
    except asyncio.CancelledError as e:
        if config.USERINPUT:
            config.USERINPUT=False
            result = input("What do you think this step should be?\n")
            result = 'any tool'
            user_input_task = asyncio.create_task(postprocess_T_generation(prediction_task, mismatch_state, collector, result, total_step_number, tas, sas, target_tasks, printed_ids=printed_ids, current_step=current_step, logger=logger, target_logger=target_logger, prev_steps=prev_steps))
            target_tasks.append(user_input_task)
    except asyncio.exceptions.TimeoutError:
        if config.USERINPUT:
            config.USERINPUT=False
            result = input("What do you think this step should be?\n")
            result = 'any tool'
            user_input_task = asyncio.create_task(postprocess_T_generation(prediction_task, mismatch_state, collector, result, total_step_number, tas, sas, target_tasks, printed_ids=printed_ids, current_step=current_step, logger=logger, target_logger=target_logger, prev_steps=prev_steps))
            target_tasks.append(user_input_task)
    except Exception as e:
        if config.USERINPUT:
            config.USERINPUT=False
            result = input("What do you think this step should be?\n")
            result = 'any tool'
            user_input_task = asyncio.create_task(postprocess_T_generation(prediction_task, mismatch_state, collector, result, total_step_number, tas, sas, target_tasks, printed_ids=printed_ids, current_step=current_step, logger=logger, target_logger=target_logger, prev_steps=prev_steps))
            target_tasks.append(user_input_task)

    end = time.time()
    t_time = round(end-start, 2)
    config.TARGET_TOTAL_TIME[total_step_number+1] = t_time

    target_logger.log(f"Intermediate Target Step {total_step_number+1} - {result} -gen {config.TARGET_TOTAL_GENERATION[total_step_number+1]} -prompt {config.TARGET_TOTAL_PROMPT[total_step_number+1]}")
    target_logger.log(f'Target: Step {total_step_number+1} -time '+ str(t_time))

    return tas, printed_ids

async def onebreakingpoint_speculative_planning(args, mismatch_state, executor, collector, encoding, app_assistant, tar_assistant, app_prompt, tar_prompt, current_step, logger, target_logger, approximation_logger, train_logger, prev_steps):
    sas = [] # approximation
    tas = [] # target
    target_tasks = []
    printed_ids = []
    mismatch_state.mismatch_detected.clear()
    pred_k = 1 if args.pred else args.k
    first = True
    
    
    # wait if the training thread is paused for debug
    # await asyncio.get_running_loop().run_in_executor(None, executor.debug_barrier.wait)
    
    i = 0
    prediction_task = None
    while i < pred_k:
        if args.pred and first: # parallel launch predictor
            prediction_task = asyncio.create_task(executor.async_predict(approximation_logger))
        
        break_out_approximation = False

        tas.append([])
        printed_ids.append([])
        approximation = asyncio.create_task(A_generate(collector, encoding, app_assistant, app_prompt, current_step+i, logger=logger, approximation_logger=approximation_logger), name=f"approximation_{current_step+i}")
        target = asyncio.create_task(T_generate(args, prediction_task, mismatch_state, collector, encoding, tar_assistant, tar_prompt, current_step+i, tas, sas, target_tasks=target_tasks, printed_ids=printed_ids, current_step=current_step, logger=logger, target_logger=target_logger, prev_steps=prev_steps), name=f"target_{i}")
        target_tasks.append(target)

        concurrent_api_calls = concurrent_calls()
        if concurrent_api_calls >= config.MAX_CONCURRENT_CALLS:
            config.MAX_CONCURRENT_CALLS = concurrent_api_calls

        try:
            sa = await approximation
            sas.append(sa)

            # check if we need to print out the approximation result here
            # which is when previous steps of target agent are all done
            flatten_tas = []
            for t in tas[:len(sas)-1]:
                if t:
                    flatten_tas += t
            flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
            flatten_ids = [t[0] for t in flatten_tas]
            flatten_tas_action = [t[1] for t in flatten_tas]
            if flatten_ids == list(range(len(flatten_ids))) and len(flatten_ids) == len(sas)-1 and all([judge_to_be_true(s, t) for s, t in zip(sas[:-1], flatten_tas_action)]):
                flattened_printed_ids = [printed_id[0] for printed_id in printed_ids if printed_id != []]
                
                if flattened_printed_ids:
                    if len(sas) == len(flattened_printed_ids)+1:
                        logger.log(f'in breaking, The approximation agent thinks step {current_step+i+1} should be ' + sa)
                        config.HIL_INTERACTION = len(sas)-1
                        register_async_handler(target_tasks=target_tasks)
                else:
                    logger.log(f'in breaking, The approximation agent thinks step {current_step+i+1} should be ' + sa)
                    config.HIL_INTERACTION = len(sas)-1
                    register_async_handler(target_tasks=target_tasks)

            # modify the prompt based on latest approximation result
            if '## Current Action Trajectory:' not in app_prompt:
                app_prompt += f'\n\n## Current Action Trajectory:\n'
            app_prompt += f'\nAction {current_step+i+1}: {str(sa)}.'
            if '## Current Action Trajectory:' not in tar_prompt:
                tar_prompt += f'\n\n## Current Action Trajectory:\n'
            tar_prompt += f'\nAction {current_step+i+1}: {str(sa)}.'
        except asyncio.CancelledError as e:
            target_logger.log("exception happens.")
            pass

        # if sa == terminate, and ta == terminate, we break the loop
        # if sa == terminate, and ta != terminate, we also break the loop
        # thus as long as sa == terminate, we break the loop
        if sa.lower() == 'terminate':
            break_out_approximation = True

        # tas is now a list of lists, so we need to flatten it in order to compare with sas
        flatten_tas = []
        for t in tas:
            if t:
                flatten_tas += t
        flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
        # halt the ongoing approximation loop
        for t in flatten_tas:
            if len(sas) > t[0]:
                if not judge_to_be_true(sas[t[0]], t[1]):
                    # print('break out of the approximation loop')
                    # print('sas', sas)
                    # print('tas', tas)
                    break_out_approximation = True
                    mismatch_state.mismatch_detected.set()
                    mismatch_state.mismatch_step_id = t[0]
                    for process_id, one_task in enumerate(target_tasks):
                        if not one_task.cancelled() and not one_task.done() and process_id > t[0]:
                            # cancel ongoing target task after mismatch
                            target_logger.log(f'Cancel Task {len(prev_steps)+process_id+1}')
                            await cancel(one_task)
                            # print('finish cancel tasks within approximation loop with task id with', process_id)
                    break

        if break_out_approximation:
            break
        
        if args.pred and first:
            pred_k = await prediction_task
            pred_k = max(pred_k, 0)
            first = False
        i += 1

    # print('====finish approximation loop====')
    # after halting the approximation loop
    # we need to collect the target results
    # organize to sas, see how much we want to preserve
    # SHOULD NOT exclude finished tasks, because exceptions are only thrown when tasks are finished
    # breakpoint()
    
    await mismatch_state.mismatch_detected.wait()
    target_logger.log(f"Mismatch at {mismatch_state.mismatch_step_id}. Cancel starts. ")
    # target_logger.log(f"target tasks: {target_tasks}")
    for process_id, one_task in enumerate(target_tasks):
        if not one_task.cancelled() and not one_task.done() and process_id > mismatch_state.mismatch_step_id:
            # cancel ongoing target task after mismatch
            target_logger.log(f'Cancel task {len(prev_steps)+process_id+1}')
            await cancel(one_task)

    # wait for pending target tasks prior to mismatch target task
    pending_tasks = [t for t in target_tasks if not t.cancelled()]
    # target_logger.log(f"pending tasks {pending_tasks}")
    
    while pending_tasks:
        break_while_loop = False
        try:
            # print('[pending_task.done() for pending_task in pending_tasks]', str([pending_task.done() for pending_task in pending_tasks]))
            if [pending_task.done() for pending_task in pending_tasks] == [True]*len(pending_tasks):
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
            # print('get to the exception part')
            if str(e) == 'terminate the whole process!':
                target_logger.log(f"Exception msg: {e}")
                # cancel all pending tasks because we have already got TERMINATE
                for process_id, one_task in pending_tasks:
                    if not one_task.cancelled() and not one_task.done() and process_id > mismatch_state.mismatch_step_id:
                        await cancel(one_task)
                # organize the results and return the final results
                flatten_tas = []
                for t in tas:
                    if t:
                        flatten_tas += t
                flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
                printed_ids, tas = simulate_leftover_interaction(collector, sas, tas, flatten_tas, printed_ids, logger, target_logger, prev_steps, target_tasks)

                # get the final tas result
                flatten_tas = []
                for t in tas:
                    if t:
                        flatten_tas += t
                flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
                for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
                    if t[0] == step_number and not judge_to_be_true(s, t[1]):
                        sas = sas[:step_number]+[flatten_tas[step_number][1]]
                        break
                target_logger.log(f"tar {step_number}: {flatten_tas[step_number][1]}")
                collector.build_trajectory(target_logger) # collector build trajectory when target terminates
                if args.pred:
                    asyncio.create_task(executor.async_train(train_logger))
                return sas
            else:
                # print('need to cancel some tasks')
                # cancel t_j for j > i if t_i != s_i
                if [pending_task.done() for pending_task in pending_tasks] == [True]*len(pending_tasks):
                    # print('no task left to be cancelled')
                    break_while_loop = True
                    # to handle all exceptions in finished tasks to avoid "exception not handled" error
                    try:
                        await asyncio.gather(*pending_tasks, return_exceptions=False)
                        break
                    except:
                        break
                if break_while_loop:
                    break
                mistaken_process_id = int(str(e)[-1])
                # print('mistaken_process_id', mistaken_process_id)
                # for process_id, one_task in enumerate(pending_tasks):
                #     if not one_task.cancelled() and not one_task.done() and process_id > mistaken_process_id:
                #         print('should cancel task id with', process_id)
                # only cancel t_j such that j > i
                for process_id, one_task in enumerate(pending_tasks):
                    if not one_task.cancelled() and not one_task.done() and process_id > mistaken_process_id:
                        # cancel pending task after mismatch
                        target_logger.log(f'Cancel task {process_id}')
                        await cancel(one_task)
                
                pending_tasks = [t for process_id, t in enumerate(target_tasks) if not t.cancelled() and process_id != mistaken_process_id]

        if break_while_loop:
            break

    # get user input or interruption
    flatten_tas = []
    for t in tas:
        if t:
            flatten_tas += t
    flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
    printed_ids, tas = simulate_leftover_interaction(collector, sas, tas, flatten_tas, printed_ids, logger, target_logger, prev_steps, target_tasks)
    
    # get the final tas result
    flatten_tas = []
    for t in tas:
        if t:
            flatten_tas += t
    flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
    mismatch = False
    origin_sa = None
    target_logger.log(f"sas tasks: {sas}")    
    target_logger.log(f"flatten_tas: {flatten_tas}")    

    for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
        if t[0] == step_number and not judge_to_be_true(s, t[1]):# t[1] != s:
            target_logger.log(f"step {step_number} app: {s}, tar: {t[1]}")
            collector.build_trajectory(target_logger) # collector build trajectory at mismatch step
            if args.pred:
                asyncio.create_task(executor.async_train(train_logger))
            sas = sas[:step_number]+[flatten_tas[step_number][1]]
            origin_sa = s
            mismatch = True
            break
    
    target_logger.log(f"return sas tasks: {sas}")    
    return sas, origin_sa, mismatch


async def speculative_planning(args, executor, encoding, app_assistant, tar_assistant, prompt, logger, target_logger, approximation_logger):
    train_logger = None
    if args.pred:
        pred_type = "dyn_k" if args.pred == True else "fix_k" 
        train_logger = Logger(f'test/logs/{args.target_type}/{pred_type}/train_datapoint{args.task_id}.log', on=True)
        # train_event = asyncio.Event()
        # train_event.clear()
        # online_train_task = asyncio.create_task(executor.async_train(train_logger))

    begin_time = datetime.now()
    collector = executor.collector
    mismatch_state = SharedState()
    steps = []
    breaking_points = 0
    i = 0
    app_prompt = prompt
    tar_prompt = prompt
    while True:
        # if args.pred == True:
        #     pred_k, state = executor.predict()
        #     # approximation_logger.log("Predict Prompt: " + state)
        # else:
        #     pred_k = args.k
        # launch async predictor training task
        # asyncio.create_task(executor.async_train(), name="predictor_online_training")
        
        # approximation_logger.log(f"Predict K: {pred_k}")
        # pred_k = 1 if pred_k == 0 else pred_k
        result, origin_sa, mismatch = await onebreakingpoint_speculative_planning(args, mismatch_state, executor, collector, encoding, app_assistant, tar_assistant, app_prompt, tar_prompt, len(steps), logger, target_logger, approximation_logger, train_logger, prev_steps=steps)
        
        
        # update approximation prompt
        if '## Current Action Trajectory:' not in app_prompt:
            app_prompt += f'\n\n## Current Action Trajectory:\n'
        previous_action_trajectory = [f'\nAction {len(steps) + j+1}: {result[j]}. Your prediction aligned with Target.' for j in range(len(result))]
        if mismatch:
            app_prompt += ''.join(previous_action_trajectory[:-1])
            app_prompt += f"\nAction {len(steps) + len(result)}: {result[-1]}. Target predicted {result[-1]} and corrected your prediction {origin_sa}."
        else: app_prompt += ''.join(previous_action_trajectory)

        # update target prompt
        if '## Current Action Trajectory:' not in tar_prompt:
            tar_prompt += f'\n\n## Current Action Trajectory:\n'
        previous_action_trajectory = [f'\nAction {len(steps) + j+1}: {result[j]}.' for j in range(len(result))]
        tar_prompt += ''.join(previous_action_trajectory)

        steps += result
        breaking_points += 1
        # logger.log('the number of breaking points: ' + str(breaking_points))
        i += len(result)

        # if the last action is terminate, we break the generation process
        if result[-1].lower() == 'terminate':
            break

    end_time = datetime.now()
    logger.log(f'{end_time} - {begin_time} = {end_time - begin_time}')
    config.TOTAL_TIME = round((end_time - begin_time).total_seconds(), 2)
    # if args.pred:
    #     train_event.set()
    #     await online_train_task
    return steps



if __name__ == '__main__':
    os.environ['OPENAI_API_KEY'] = "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"
    encoding = tiktoken.get_encoding("cl100k_base")
    
    asyncio.get_event_loop().set_debug(True)
    
    ## gloabl variables
    config.MAX_CONCURRENT_CALLS = 0
    config.TOTAL_APPROXIMATION_CALLS = 0
    config.TOTAL_CORRECT_APPROXIMATION_CALLS = 0
    config.TOTAL_TOKEN_GENERATION = 0
    config.TOTAL_TOKEN_PROMPT = 0
    config.USERINPUT=False
    config.TARGET_TOTAL_TIME = {}
    config.TARGET_TOTAL_GENERATION = {}
    config.APPROX_TOTAL_GENERATION = 0
    config.TOTAL_TIME = 0
    config.TERMINATE_TARGET_ID = None

    config.APPROX_TOTAL_PROMPT = 0
    config.TARGET_TOTAL_PROMPT = {}
    config.APPROXIMATION_TOTAL_PROMPT = 0

    random.seed(2)
    parser = argparse.ArgumentParser(description='OpenAGI')
    parser.add_argument('--data', type=str, default='data/openagi_task_descrition.txt', help='data directory')
    parser.add_argument('--task_id', type=int, default=12, help='task id')
    parser.add_argument('--k', type=int, default=2, help='number of approximation steps to generate everytime')
    parser.add_argument('--target_type', type=str, default='react', help='cot, react, multi-agent, direct')
    parser.add_argument('--pred', type=bool, default=False, help='speculative planning with predictor')
    
    args = parser.parse_args()
    pred_type = "dyn_k" if args.pred == True else "fix_k" 
    logger = Logger(f'4_7/logs/{args.target_type}/{pred_type}/simulation_datapoint{args.task_id}.log', on=True)
    target_logger = Logger(f'4_7/logs/{args.target_type}/{pred_type}/target_datapoint{args.task_id}.log', on=True)
    approximation_logger = Logger(f'4_7/logs/{args.target_type}/{pred_type}/approximation_datapoint{args.task_id}.log', on=True)

    tasks = load_data(args)
    task_description = tasks[args.task_id]
    logger.log('task description: ' + task_description)
    target_logger.log('task description: ' + task_description)
    approximation_logger.log('task description: ' + task_description)
    tools = """
Available tools are as follows:

(1) <tool>Sentiment Analysis</tool>
(2) <tool>Text Summarization</tool>
(3) <tool>Machine Translation</tool>
(4) <tool>Fill Mask</tool>
(5) <tool>Question Answering</tool>
(6) <tool>Image Classification</tool>
(7) <tool>Object Detection</tool>
(8) <tool>Colorization</tool>
(9) <tool>Image Super-Resolution</tool>
(10) <tool>Image Denoising</tool>
(11) <tool>Image Deblurring</tool>
(12) <tool>Visual Question Answering</tool>
(13) <tool>Image Captioning</tool>
(14) <tool>Text-to-Image Generation</tool>
(15) <tool>TERMINATE</tool>

For each step of the plan, please specify the tool you would like to use. But if you think the task is completed, please use <tool>TERMINATE</tool> to end the conversation.

Please use xml tags to specify the tool when responsing. For example, <tool>Sentiment Analysis</tool> for Sentiment Analysis.
"""

    prompt = "## Problem: " + task_description + "\nPlease solve this problem using the following tools step by step:\n" + tools

    if args.target_type == 'direct':
        app_config_list = [{
            "model": "gpt-3.5-turbo",
            # "api_key": "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",
            "api_key": os.environ['OPENAI_API_KEY'],
            "api_type": "openai",
            "cache_seed": None, 
            "temperature": 0,
            "seed":0
        },]
    else:
        app_config_list = [{
            "model": "gpt-4-turbo",
            # "model": "gpt-3.5-turbo",
            "api_key": os.environ['OPENAI_API_KEY'],
            # "api_key": "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",
            "api_type": "openai",
            "cache_seed": None,
            "temperature": 0,
            "seed":0
        },]
    app_assistant = AssistantAgent("assistant", llm_config={"config_list": app_config_list}, human_input_mode='NEVER')

    
    tar_config_list = [{
        "model": "gpt-4-turbo",
        # "model": "gpt-3.5-turbo",
        "api_key": os.environ['OPENAI_API_KEY'],
        # "api_key": "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",
        "api_type": "openai",
        "cache_seed": None, 
        "temperature": 0,
        "seed":0
    },]
    tar_assistant = AssistantAgent("assistant", llm_config={"config_list": tar_config_list}, human_input_mode='NEVER')
    print(app_assistant.llm_config)
    print(tar_assistant.llm_config)
    # online learning preparations
    model_path = "../weights/distilbert-base-uncased"
    bert_model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DistilBERTValueFunction(bert_model)
    
    model_save_path = "cot_value_function_model.pth"
    model.load_state_dict(torch.load(model_save_path))


    executor = OnlineLearningExecutor(
        model=model,
        tokenizer=tokenizer,
        initial_prompt=task_description,
        buffer_size=1000,
        batch_size_steps=4,
        lambda_=0.9,
        gamma=1,
        lr=5e-05,
        epoch_per_train=3
    )
    # checkpoint = torch.load('predictor_online_weights.pth')
    # executor.predict_model.load_state_dict(checkpoint['model_state_dict'])
    # executor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # run the speculative planning
    steps = asyncio.run(speculative_planning(args, executor, encoding, app_assistant, tar_assistant, prompt, logger, target_logger, approximation_logger))

    # record the metrics
    logger.log('final result for the speculative planning ' + str(steps))
    logger.log('max concurrent calls: ' + str(config.MAX_CONCURRENT_CALLS-1)) # speculative_planning will add one more call
    sp_plan_token = config.TOTAL_TOKEN_GENERATION
    logger.log('total approximation token generation: ' + str(config.APPROX_TOTAL_GENERATION))
    step_num = len(steps)

    naive_plan_token = sum(config.TARGET_TOTAL_GENERATION[i] for i in range(1, step_num+1))
    naive_plan_time = sum(config.TARGET_TOTAL_TIME[i] for i in range(1, step_num+1))
    naive_plan_prompt = sum(config.TARGET_TOTAL_PROMPT[i] for i in range(1, step_num+1))
    logger.log('total target token prompt: ' + str(naive_plan_prompt))
    logger.log('total target token generation: ' + str(naive_plan_token))
    logger.log('total target step time: ' + str(naive_plan_time))
    logger.log('total token prompt: ' + str(config.TOTAL_TOKEN_PROMPT))
    logger.log('total token generation: ' + str(sp_plan_token))

    logger.log('accuracy of approximation agent: ' + str(config.TOTAL_CORRECT_APPROXIMATION_CALLS/config.TOTAL_APPROXIMATION_CALLS))
    avg_sp_token = round(sp_plan_token/step_num, 2)
    logger.log('sp token/step: ' + str(avg_sp_token))
    avg_naive_token = round(naive_plan_token/step_num, 2)
    logger.log('naive token/step: ' + str(avg_naive_token))
    logger.log('generation token cost increased: ' + str(round((avg_sp_token/avg_naive_token-1)*100, 2))+"%.")
    
    avg_sp_prompt = config.TOTAL_TOKEN_PROMPT / step_num
    logger.log('sp prompt/step: ' + str(round(avg_sp_prompt, 2)))

    avg_naive_prompt = round(naive_plan_prompt/step_num, 2)
    logger.log('naive prompt/step: ' + str(avg_naive_prompt))
    logger.log('prompt token cost increased: ' + str(round((avg_sp_prompt/avg_naive_prompt-1)*100, 2))+"%.")

    avg_sp_time = round(config.TOTAL_TIME/step_num, 2)
    logger.log('sp time/step: ' + str(avg_sp_time))
    avg_naive_time = round(naive_plan_time/step_num, 2)
    logger.log('naive time/step: ' + str(avg_naive_time))
    logger.log('time decreased by: ' + str(round((1-avg_sp_time/avg_naive_time)*100, 2))+"%.")
    logger.log(f'step number: {step_num}')

    if not args.pred:
        logger.log(f'k = {args.k}')
    else: logger.log(f'dynamic k')
    torch.save({
        'model_state_dict': executor.predict_model.state_dict(),
        'optimizer_state_dict': executor.optimizer.state_dict()
    }, 'predictor_online_weights.pth')
    print("model saved to predictor_online_weights.pth")

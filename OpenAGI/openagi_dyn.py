import os
import config
import asyncio
import time
import random
import json 
import torch
import argparse 
# from color import slow_type_approximation, slow_type_target
import autogen 
import tiktoken
import openai
from openai import OpenAI
from datetime import datetime
from autogen import AssistantAgent
from util import Logger, cancel, register_async_handler

from transformers import AutoModel, AutoTokenizer
from predictor import DistilBERTValueFunction
from OpenAGI.async_online_utils import OnlineLearningExecutor, SharedState


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

async def A_generate(args, collector, encoding, assistant, prompt, total_step_number, logger, approximation_logger, start):
    if args.approx_type == "direct":
        prompt += f"\n\nDirectly tell me what **the ONE NEXT action step** based on the current action trajectory should be. (Remember to use xml tag <tool> and </tool> for formatting.)\nWhat should be the action in Step {total_step_number+1}?\nTarget won't terminate easily so don't choose <tool>TERMINATE</tool> unless you are very confident. \nStep {total_step_number+1}:"
    else: # cot
        prompt += f"\n\nCarefully think about **the ONE NEXT action step** based on the current action trajectory, by first providing a clear reasoning chain. And then decide which tool to use for the current step. (Remember to use xml tag <tool> and </tool> for formatting.)\nWhat should be the action in Step {total_step_number+1}?\nTarget won't terminate easily so don't choose <tool>TERMINATE</tool> unless you are very confident. \nStep {total_step_number+1}:"

    n = 0

    while True:
        try:
            n += 1
            if n >= 10:
                result = ''
                return result
            prompt_token = len(encoding.encode(prompt))
            config.TOTAL_TOKEN_PROMPT += prompt_token
            config.APPROX_SP_PROMPT += prompt_token
            config.APPROX_NORMAL_PROMPT[total_step_number+1] = prompt_token
            approximation_logger.log(f'Approximation: Step {total_step_number+1} -prompt token {prompt_token}')
            response = await assistant.a_generate_reply(messages=[{'content':prompt, 'role':'user'}])
            
            app_tokens = len(encoding.encode(response))
            config.TOTAL_TOKEN_GENERATION += app_tokens
            config.APPROX_SP_GENERATION += app_tokens
            config.APPROX_NORMAL_GENERATION[total_step_number+1] = app_tokens
            
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
            config.APPROX_NROMAL_TIME[total_step_number+1] = a_time
            
            return result
        except:
            continue


async def MultiAgent(encoding, assistants, prompt, total_step_number, target_logger):
    # first round of multi-agent discussion
    
    prompt4A = prompt + f"\n\nYou will discuss with another agent about **the ONE NEXT action step** based on the current action trajectory. Please provide your thought and answer first.\nAction {total_step_number}:"
    prompt4A_token = len(encoding.encode(prompt4A))
    config.TARGET_NORMAL_PROMPT[total_step_number+1] = prompt4A_token
    config.TOTAL_TOKEN_PROMPT += prompt4A_token
    target_logger.log(f"Target step {total_step_number+1} thought A prompt token: {prompt4A_token}")

    thoughtA = await assistants[0].a_generate_reply(messages=[{'content':prompt4A, 'role':'user'}])
    if type(thoughtA) != str:
        thoughtA = thoughtA['content']

    thoughtA_token = len(encoding.encode(thoughtA))
    config.TOTAL_TOKEN_GENERATION += thoughtA_token
    config.TARGET_NORMAL_GENERATION[total_step_number+1] = thoughtA_token
    target_logger.log(f"Target step {total_step_number+1} thought A generation token: {thoughtA_token}")

    prompt4B = prompt + f"\n\nYou are discussing with another agent about **the ONE NEXT action step** based on the current action trajectory.\nThe other agent's idea about this step is {thoughtA}.\nPlease think about whether the other agent's thought and idea is useful, and then provide your thought and answer now.\nAction {total_step_number}:"
    prompt4B_token = len(encoding.encode(prompt4B))
    config.TARGET_NORMAL_PROMPT[total_step_number+1] += prompt4B_token
    config.TOTAL_TOKEN_PROMPT += prompt4B_token
    target_logger.log(f"Target step {total_step_number+1} thought B prompt token: {prompt4B_token}")

    thoughtB = await assistants[1].a_generate_reply(messages=[{'content':prompt4B, 'role':'user'}])
    if type(thoughtB) != str:
        thoughtB = thoughtB['content']

    thoughtB_token = len(encoding.encode(thoughtB))
    config.TOTAL_TOKEN_GENERATION += thoughtB_token
    config.TARGET_NORMAL_GENERATION[total_step_number+1] += thoughtB_token
    target_logger.log(f"Target step {total_step_number+1} thought B generation token: {thoughtB_token}")
    
    # second round of multi-agent discussion
    prompt4A = prompt + f"\n\nYou are discussing with another agent about **the ONE NEXT action step** based on the current action trajectory.\nYour original thought and idea about this step is {thoughtA}.\nThe other agent's thought and idea about this step is {thoughtB}.\nPlease summarize and reflect, and then update your thought on what this step should be after updating.\nAction {total_step_number}:"
    prompt4A_token = len(encoding.encode(prompt4A))
    config.TARGET_NORMAL_PROMPT[total_step_number+1] += prompt4A_token
    config.TOTAL_TOKEN_PROMPT += prompt4A_token
    target_logger.log(f"Target step {total_step_number+1} thought A prompt token: {prompt4A_token}")

    thoughtA = await assistants[0].a_generate_reply(messages=[{'content':prompt4A, 'role':'user'}])
    if type(thoughtA) != str:
        thoughtA = thoughtA['content']
    
    thoughtA_token = len(encoding.encode(thoughtA))
    config.TOTAL_TOKEN_GENERATION += thoughtA_token
    config.TARGET_NORMAL_GENERATION[total_step_number+1] += thoughtA_token
    target_logger.log(f"Target step {total_step_number+1} thought A generation token: {thoughtA_token}")

    prompt4B = prompt + f"\n\nYou are discussing with another agent about **the ONE NEXT action step** based on the current action trajectory.\nYour original thought and idea about this step is {thoughtB}.\nThe other agent's thought and idea about this step is {thoughtA}.\nPlease summarize and reflect, and then update your thought on what this step should be after updating.\nAction {total_step_number}:"
    prompt4B_token = len(encoding.encode(prompt4B))
    config.TARGET_NORMAL_PROMPT[total_step_number+1] += prompt4B_token
    config.TOTAL_TOKEN_PROMPT += prompt4B_token
    target_logger.log(f"Target step {total_step_number+1} thought B prompt token: {prompt4B_token}")

    thoughtB = await assistants[1].a_generate_reply(messages=[{'content':prompt4B, 'role':'user'}])
    if type(thoughtB) is not str:
        thoughtB = thoughtB['content']
    
    thoughtB_token = len(encoding.encode(thoughtB))
    config.TOTAL_TOKEN_GENERATION += thoughtB_token
    config.TARGET_NORMAL_GENERATION[total_step_number+1] += thoughtB_token
    target_logger.log(f"Target step {total_step_number+1} thought B generation token: {thoughtB_token}")
    
    
    # generate action based on thought
    round2prompt4A = prompt + f"\n\nYou are discussing with another agent about **the ONE NEXT action step** based on the current action trajectory.\nYour original thought and idea about this step is {thoughtA}.\nThe other agent's thought and idea about this step is {thoughtB}.\nPlease summarize and reflect, and then provide your thought and answer for what this step should be.\nAction {total_step_number}:"
    r2prompt4A_token = len(encoding.encode(round2prompt4A))
    config.TARGET_NORMAL_PROMPT[total_step_number+1] += r2prompt4A_token
    config.TOTAL_TOKEN_PROMPT += r2prompt4A_token
    target_logger.log(f"Target step {total_step_number+1} round 2 thought A prompt token: {r2prompt4A_token}")

    response = await assistants[0].a_generate_reply(messages=[{'content':round2prompt4A, 'role':'user'}])
    if type(response) != str:
        response = response['content']
    
    response_token = len(encoding.encode(response))
    config.TOTAL_TOKEN_GENERATION += response_token
    config.TARGET_NORMAL_GENERATION[total_step_number+1] += response_token
    target_logger.log(f"Target step {total_step_number+1} response generation token: {response_token}")
    return response


async def ReAct(encoding, assistant, prompt, total_step_number, target_logger):
    prompt += f"\n\nCarefully think about **the ONE NEXT action step** based on the current action trajectory."
    prompt += f"\nGenerate thought only.\nThought {total_step_number}:"

    target_logger.log('react launch api for thought.')

    prompt_token = len(encoding.encode(prompt))
    config.TARGET_NORMAL_PROMPT[total_step_number+1] = prompt_token
    config.TOTAL_TOKEN_PROMPT += prompt_token
    target_logger.log(f"Target step {total_step_number+1} thought prompt token: {prompt_token}")

    thought = await assistant.a_generate_reply(messages=[{'content':prompt, 'role':'user'}])

    thought_token = len(encoding.encode(thought))
    config.TOTAL_TOKEN_GENERATION += thought_token
    config.TARGET_NORMAL_GENERATION[total_step_number+1] = thought_token
    target_logger.log(f"Target step {total_step_number+1} thought generation token: {thought_token}")

    prompt += " " + thought
    prompt += f"\nGenerate Action only based on thoughts. Remember to use xml tag <tool> and </tool> for formatting. \nAction {total_step_number}:"
    # generate action based on thought
    target_logger.log('react launch api for response.')
    prompt_token = len(encoding.encode(prompt))
    config.TARGET_NORMAL_PROMPT[total_step_number+1] += prompt_token
    config.TOTAL_TOKEN_PROMPT += prompt_token
    target_logger.log(f"Target step {total_step_number+1} response prompt token: {prompt_token}")
    response = await assistant.a_generate_reply(messages=[{'content':prompt, 'role':'user'}])
    response_token = len(encoding.encode(response))
    config.TOTAL_TOKEN_GENERATION += response_token
    config.TARGET_NORMAL_GENERATION[total_step_number+1] += response_token
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
            target_logger.log(f"step number:{step_number}, t[0]: {t[0]}, t[1]: {t[1]}")
            if t[1].lower() == 'terminate':
                end = time.time()
                if not mismatch_state.mismatch_detected.is_set():
                    mismatch_state.mismatch_step_id = t[0]
                    mismatch_state.mismatch_detected.set()
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
                if not mismatch_state.mismatch_detected.is_set():
                    mismatch_state.mismatch_step_id = ta[0] # mismatch occurs
                    mismatch_state.mismatch_detected.set()

                # end = time.time()
                # pending_target_tasks = [t for t in asyncio.all_tasks() if not t.cancelled() and not t.done() and t in target_tasks and t.get_name().startswith('target')]
                # for pending_target_task in pending_target_tasks:
                #     # mismatch occurs, cancel ongoing approximation tasks
                #     await cancel(pending_target_task)
                
                pending_approximation_tasks = [t for t in asyncio.all_tasks() if not t.cancelled() and not t.done() and t not in target_tasks and t.get_name().startswith('approximation')]
                for pending_approximation_task in pending_approximation_tasks:
                    # mismatch occurs, cancel ongoing approximation tasks
                    await cancel(pending_approximation_task)
                raise Exception(f'approximation error happen in step {total_step_number} for current step {current_step}, the target id is {ta[0]}')
    if config.ENABLE_PRED:
        k = await prediction_task
    else: k = args.k

    if not mismatch_state.mismatch_detected.is_set() and ta[0] >= k-1:
        mismatch_state.mismatch_step_id = ta[0]
        mismatch_state.mismatch_detected.set()

    return tas, printed_ids

async def T_generate(args, prediction_task, mismatch_state, collector, encoding, assistant, prompt, total_step_number, tas, sas, target_tasks, printed_ids, current_step, logger, target_logger, prev_steps, start):
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
                elif args.target_type == 'multi_agent':
                    response = await MultiAgent(encoding, assistant, prompt, total_step_number, target_logger)
                else:
                    # if mismatch_state.mismatch_detected.is_set() and mismatch_state.mismatch_step_id < 
                    # log prompt token
                    prompt_token = len(encoding.encode(prompt))
                    config.TARGET_NORMAL_PROMPT[total_step_number+1] = prompt_token
                    config.TOTAL_TOKEN_PROMPT += prompt_token
                    response = await assistant.a_generate_reply(messages=[{'content':prompt, 'role':'user'}])
                    response_token = len(encoding.encode(response))
                    config.TOTAL_TOKEN_GENERATION += response_token
                    config.TARGET_NORMAL_GENERATION[total_step_number+1] = response_token
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
    config.TARGET_NORMAL_TIME[total_step_number+1] = t_time

    target_logger.log(f"Intermediate Target Step {total_step_number+1} - {result} -gen {config.TARGET_NORMAL_GENERATION[total_step_number+1]} -prompt {config.TARGET_NORMAL_PROMPT[total_step_number+1]}")
    target_logger.log(f'Target: Step {total_step_number+1} -time '+ str(t_time))

    return tas, printed_ids

async def onebreakingpoint_speculative_planning(args, mismatch_state, executor, collector, encoding, app_assistant, tar_assistant, app_prompt, tar_prompt, current_step, logger, target_logger, approximation_logger, prev_steps):
    sas = [] # approximation
    tas = [] # target
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
        if config.ENABLE_PRED and first: # parallel launch predictor
            prediction_task = asyncio.create_task(executor.async_predict(approximation_logger, args.offset))
        
        break_out_approximation = False

        tas.append([])
        printed_ids.append([])
        a_start = time.time()
        approximation = asyncio.create_task(A_generate(args, collector, encoding, app_assistant, app_prompt, current_step+i, logger=logger, approximation_logger=approximation_logger, start=a_start), name=f"approximation_{current_step+i}")
        t_start = time.time()
        target = asyncio.create_task(T_generate(args, prediction_task, mismatch_state, collector, encoding, tar_assistant, tar_prompt, current_step+i, tas, sas, target_tasks=target_tasks, printed_ids=printed_ids, current_step=current_step, logger=logger, target_logger=target_logger, prev_steps=prev_steps, start=t_start), name=f"target_{i}")
        target_tasks.append(target)

        concurrent_api_calls = concurrent_calls()
        if concurrent_api_calls >= config.MAX_CONCURRENT_CALLS:
            config.MAX_CONCURRENT_CALLS = concurrent_api_calls

        try:
            sa = await approximation
            sas.append(sa)
            if mismatch_state.mismatch_detected.is_set():
                break

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
                app_prompt += '\n\n## Current Action Trajectory:\n'
            app_prompt += f'\nAction {current_step+i+1}: {str(sa)}.'
            if '## Current Action Trajectory:' not in tar_prompt:
                tar_prompt += '\n\n## Current Action Trajectory:\n'
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
                if not judge_to_be_true(sas[t[0]], t[1]) or t[1].lower() == "terminate":
                    break_out_approximation = True
                    mismatch_state.mismatch_step_id = t[0]
                    mismatch_state.mismatch_detected.set()

                    for process_id, one_task in enumerate(target_tasks):
                        if not one_task.cancelled() and not one_task.done() and process_id > t[0]:
                            # cancel ongoing target task after mismatch
                            target_logger.log(f'Cancel Task {len(prev_steps)+process_id+1}')
                            await cancel(one_task)
                    break

        if break_out_approximation:
            break
        
        if config.ENABLE_PRED and first:
            pred_k = await prediction_task
            config.PREDICT_K.append(pred_k)
            config.PREDICT_TOTAL += 1
            pred_k = max(pred_k, 1)
            first = False
        i += 1

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
    
    for process_id, t in enumerate(target_tasks):
        if t.done() or t.cancelled():
            continue
        try:
            await t
            flatten_tas = []
            for t in tas:
                if t:
                    flatten_tas += t
            flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
            for t in flatten_tas:
                if len(sas) > t[0]:
                    if not judge_to_be_true(sas[t[0]], t[1]):
                        for process_id, one_task in enumerate(pending_tasks):
                            if not one_task.cancelled() and not one_task.done() and process_id > t[0]:
                                # cancel ongoing target task after mismatch
                                target_logger.log(f'Cancel task: {len(prev_steps)+process_id+1}')
                                await cancel(one_task)
                        break
        except Exception as e:
            print(f"An error occurred with task {len(prev_steps)+t[0]+1}: {e}")


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
        if t[0] == step_number and (not judge_to_be_true(s, t[1]) or t[1].lower() == "terminate"):# t[1] != s:
            target_logger.log(f"step {step_number} app: {s}, tar: {t[1]}")
            config.PREDICT_CORRECT += collector.build_trajectory(target_logger, config.PREDICT_K) # collector build trajectory at mismatch step
            config.PREDICT_K = [] 

            if config.ENABLE_TRAIN:
                if config.BUILD_TRAJ_TIMES == 0:
                    asyncio.create_task(executor.async_train())
                config.BUILD_TRAJ_TIMES = (config.BUILD_TRAJ_TIMES + 1) % config.TRAIN_INTERVAL
            sas = sas[:step_number]+[flatten_tas[step_number][1]]
            origin_sa = s
            mismatch = True
            break
    
    return sas, origin_sa, mismatch


async def speculative_planning(args, executor, encoding, app_assistant, tar_assistant, prompt, logger, target_logger, approximation_logger):
        
    begin_time = datetime.now()
    collector = executor.collector
    mismatch_state = SharedState()
    await mismatch_state.initialize()
    steps = []
    breaking_points = 0
    i = 0

    app_prompt = prompt
    tar_prompt = prompt
    while True:
        result, origin_sa, mismatch = await onebreakingpoint_speculative_planning(args, mismatch_state, executor, collector, encoding, app_assistant, tar_assistant, app_prompt, tar_prompt, len(steps), logger, target_logger, approximation_logger, prev_steps=steps)
        
        # update approximation prompt
        if '## Current Action Trajectory:' not in app_prompt:
            app_prompt += '\n\n## Current Action Trajectory:\n'
        previous_action_trajectory = [f'\nAction {len(steps) + j+1}: {result[j]}. Your prediction aligned with Target.' for j in range(len(result))]
        if mismatch:
            app_prompt += ''.join(previous_action_trajectory[:-1])
            app_prompt += f"\nAction {len(steps) + len(result)}: {result[-1]}. Target predicted {result[-1]} and corrected your prediction {origin_sa}."
        else: app_prompt += ''.join(previous_action_trajectory)

        # update target prompt
        if '## Current Action Trajectory:' not in tar_prompt:
            tar_prompt += '\n\n## Current Action Trajectory:\n'
        previous_action_trajectory = [f'\nAction {len(steps) + j+1}: {result[j]}.' for j in range(len(result))]
        tar_prompt += ''.join(previous_action_trajectory)

        steps += result
        breaking_points += 1
        i += len(result)

        # if the last action is terminate, we break the generation process
        if result[-1].lower() == 'terminate' or len(steps) >= config.MAX_STEP:
            break

    end_time = datetime.now()
    logger.log(f'{end_time} - {begin_time} = {end_time - begin_time}')
    config.TOTAL_SP_TIME = round((end_time - begin_time).total_seconds(), 2)
    
    return steps


def run_one_task(args, task_id, executor, encoding, app_assistant, tar_assistant, traj_file):
    ## gloabl variables
    config.MAX_CONCURRENT_CALLS = 0
    config.TOTAL_APPROXIMATION_CALLS = 0
    config.TOTAL_CORRECT_APPROXIMATION_CALLS = 0

    config.TOTAL_TOKEN_GENERATION = 0
    config.TOTAL_TOKEN_PROMPT = 0
    config.USERINPUT=False

    config.TARGET_NORMAL_PROMPT = {}
    config.TARGET_NORMAL_GENERATION = {}

    config.APPROX_SP_PROMPT = 0
    config.APPROX_SP_GENERATION = 0
    config.APPROX_NORMAL_PROMPT = {}
    config.APPROX_NORMAL_GENERATION = {}

    config.TOTAL_SP_TIME = 0
    config.TARGET_NORMAL_TIME = {}
    config.APPROX_NROMAL_TIME = {}

    config.PREDICT_K = []
    config.PREDICT_CORRECT = 0
    config.PREDICT_TOTAL = 0
    
    random.seed(2)
    pred_type = "dyn_k" if args.pred else "fix_k"

    log_dir = f"data/{args.approx_type}_{args.target_type}/{args.model_type}/{pred_type}"
    if config.ENABLE_TRAIN: 
        logger = Logger(f'{log_dir}/tau_{args.tau}_offset_{args.offset}/simulation_datapoint{task_id}.log', on=True)
        target_logger = Logger(f'{log_dir}/tau_{args.tau}_offset_{args.offset}/target_datapoint{task_id}.log', on=True)
        approximation_logger = Logger(f'{log_dir}/tau_{args.tau}_offset_{args.offset}/approximation_datapoint{task_id}.log', on=True)
    else:
        logger = Logger(f'{log_dir}/k_{args.k}/simulation_datapoint{task_id}.log', on=True)
        target_logger = Logger(f'{log_dir}/k_{args.k}/target_datapoint{task_id}.log', on=True)
        approximation_logger = Logger(f'{log_dir}/k_{args.k}/approximation_datapoint{task_id}.log', on=True)

    tasks = load_data(args)
    task_description = tasks[task_id]
    executor.set_initial_task_prompt(task_description)
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

    steps = asyncio.run(speculative_planning(args, executor, encoding, app_assistant, tar_assistant, prompt, logger, target_logger, approximation_logger))

    # record the metrics
    logger.log('final result for the speculative planning ' + str(steps))
    logger.log('max concurrent calls: ' + str(config.MAX_CONCURRENT_CALLS-1)) # speculative_planning will add one more call
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
    logger.log('accuracy of approximation agent: ' + str(config.TOTAL_CORRECT_APPROXIMATION_CALLS/config.TOTAL_APPROXIMATION_CALLS))
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
    if config.ENABLE_PRED:
        logger.log(f'predictor acc: {round(config.PREDICT_CORRECT / config.PREDICT_TOTAL, 2)}')
    logger.log(f'step number: {step_num}')

    if not args.pred:
        logger.log(f'k = {args.k}')
        traj_file = traj_file.format(task_id)
    else: 
        logger.log('dynamic k')

    executor.save_trajectory(traj_file, config.ENABLE_TRAIN)


if __name__ == '__main__':
    os.environ['DEEPSEEK_API_KEY'] = ""
    os.environ['OPENAI_API_KEY'] = ""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    parser = argparse.ArgumentParser(description='OpenAGI')
    parser.add_argument('--data', type=str, default='data/openagi_task_descrition.txt', help='data directory')
    parser.add_argument('--k', type=int, default=2, help='number of approximation steps to generate everytime')
    parser.add_argument('--approx_type', type=str, default='direct', help='cot, direct')
    parser.add_argument('--target_type', type=str, default='react', help='react, multi_agent')
    parser.add_argument('--model_type', type=str, default="gpt-4.1-mini", help='gpt-4.1-mini, deepseek')
    parser.add_argument('--pred', action='store_true', help='enable speculative planning with predictor')
    parser.add_argument('--no-pred', dest='pred', action='store_false', help='disable speculative planning with predictor')
    parser.set_defaults(pred=True)
    parser.add_argument('--lr', type=float, default=1e-5, help='online learning lr')
    parser.add_argument('--ep', type=int, default=3, help='online learning epoch per train')
    parser.add_argument('--bf', type=int, default=2500, help='online learning buffer size')
    parser.add_argument('--bs', type=int, default=16, help='online learning batch size')
    parser.add_argument('--gma', type=float, default=1, help='online learning gamma for lambda return calculation')
    parser.add_argument('--lmd', type=float, default=0.95, help='online learning lambda for lambda return calculation')
    parser.add_argument('--load', dest='load', action='store_true', help='load previous trajectory and model')
    parser.add_argument('--no-load', dest='load', action='store_false', help='do not load previous trajectory and model')
    parser.add_argument('--tau', type=float, default=0.5, help='expectile loss tau')
    parser.add_argument('--s_task', type=int, default=1, help='start task id')
    parser.add_argument('--freq', type=int, default=1, help='online learning training frequency')
    parser.add_argument('--offset', type=int, default=0, help='biased inference offset for k')
    
    parser.set_defaults(load=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config.TRAIN_INTERVAL = args.freq
    config.BUILD_TRAJ_TIMES = 0
    config.MAX_STEP = 20

    model_type = args.model_type

    if model_type == "deepseek":     
        app_config_list = [{
            "model": "deepseek-chat",
            "api_key": os.environ['DEEPSEEK_API_KEY'],
            "base_url": "https://api.deepseek.com",
            "api_type": "deepseek",
            "cache_seed": None, 
            "temperature": 0,
            "top_p": 1.0,
            "seed":0,
            
        },]
        tar_config_list = [{
            "model":  "deepseek-reasoner",
            "api_key": os.environ['DEEPSEEK_API_KEY'],
            "base_url": "https://api.deepseek.com",
            "api_type": "deepseek",
            "cache_seed": None, 
            "temperature": 0,
            "top_p": 1.0,
            "seed":0
        },]
    else:
        app_config_list = [{
            "model": model_type,
            "api_key": os.environ['OPENAI_API_KEY'],
            "api_type": "openai",
            "cache_seed": None, 
            "temperature": 0,
            "top_p": 1.0,
            "seed":0
        },]
        tar_config_list = [{
            "model":  model_type,
            "api_key": os.environ['OPENAI_API_KEY'],
            "api_type": "openai",
            "cache_seed": None,
            "temperature": 0,
            "top_p": 1.0,
            "seed":0
        },]

    app_assistant = AssistantAgent("assistant", llm_config={"config_list": app_config_list}, human_input_mode='NEVER')
    tar_assistant = AssistantAgent("assistant", llm_config={"config_list": tar_config_list}, human_input_mode='NEVER')
    if args.target_type == 'multi_agent':
        if model_type == "deepseek-chat":
            tar_config_list = [{
                "model":  "deepseek-reasoner",
                "api_key": os.environ['DEEPSEEK_API_KEY'],
                "base_url": "https://api.deepseek.com",
                "api_type": "deepseek",
                "cache_seed": None, 
                "temperature": 0,
                "top_p": 1.0,
                "seed":0
            },]
        else:
            tar_config_list = [{
                "model": "gpt-4.1-mini",
                "api_key": os.environ['OPENAI_API_KEY'],
                "api_type": "openai",
                "cache_seed": None, 
                "temperature": 0,
                "top_p": 1.0,
                "seed":0
            },]
        tar_assistantB = AssistantAgent("assistant", llm_config={"config_list": tar_config_list}, human_input_mode='NEVER')
        tar_assistant = [tar_assistant, tar_assistantB]

    # online learning preparations
    model_path = "distilbert-base-uncased"
    bert_model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DistilBERTValueFunction(bert_model).to(device)
    
    if args.pred: # dyn k
        traj_dir = f"trajectory/online_traj/{args.approx_type}_{args.target_type}/{args.model_type}"
        os.makedirs(traj_dir, exist_ok=True)
        traj_file = f"{traj_dir}/tau_{args.tau}_offset_{args.offset}.ndjson"
    else: # fix k
        traj_dir = f"trajectory/{args.approx_type}_{args.target_type}/{args.model_type}/fix_k_{args.k}"
        os.makedirs(traj_dir, exist_ok=True)
        traj_file = f"{traj_dir}/task_{{}}.json"
    
    ckpt_dir = f"ckpt/online/{args.approx_type}_{args.target_type}/{args.model_type}"
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
        tau=args.tau
    )
    

    # run the speculative planning for multiple tasks
    config.ENABLE_TRAIN = args.pred
    config.WARMUP = 0
    config.ENABLE_PRED = args.pred
    
    task_ids = list(range(args.s_task, 313))
    warmup_task = 0
    for task_id in task_ids:
        if warmup_task >= config.WARMUP:
            config.ENABLE_PRED = args.pred
        run_one_task(args, task_id, executor, encoding, app_assistant, tar_assistant, traj_file)
        warmup_task += 1
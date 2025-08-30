"""Dynamic Speculative Planning implementation for Travel Planner."""

import asyncio
import time
from datetime import datetime
from typing import List
import nltk

from config import Config
from async_online_utils import SharedState
from util import cancel, register_async_handler


def judge_to_be_true(s, t):
    """Judge if two actions are semantically equivalent."""
    try:
        approximation_function_name = s.split("[")[0].strip()
        target_function_name = t.split("[")[0].strip()

        approximation_function_arg = s[s.index("[") : s.index("]")].strip()
        target_function_arg = t[t.index("[") : t.index("]")].strip()

        def token_edit_levenstein_similarity_normalized(
            text1: str, text2: str
        ) -> float:
            """Compute the normalized levenstein distance between two texts."""
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
    except Exception:
        if s == t:
            return True
        else:
            return False


def concurrent_calls():
    """Get the number of concurrent API calls."""
    tasks = asyncio.all_tasks()
    pending_tasks = [t for t in tasks if not t.done() and not t.cancelled()]
    return len(pending_tasks)


def interaction_function(s, t, logger, collector, previous_steps, config):
    """Handle interaction between approximation and target agents."""
    cur_time = time.time()
    timestamp = datetime.fromtimestamp(cur_time)
    source = "Target"
    step = t[0][0] + len(previous_steps) + 1
    desc = t[0][1][0]

    collector.record_step(timestamp, source, step, desc)
    
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

    return desc


class TravelPlannerSpeculativePlanner:
    def __init__(self, collector, encoding, executor, app_agent, tar_agent, config: Config):
        self.collector = collector
        self.encoding  = encoding
        self.executor  = executor
        self.app_agent = app_agent
        self.tar_agent = tar_agent
        self.config    = config
        self.mismatch_state = SharedState()
        self.steps = []
        
        self.logger = None
        self.target_logger = None
        self.approximation_logger = None

    def set_loggers(self, logger, target_logger, approximation_logger, train_logger=None):
        """Set loggers for the planner."""
        self.logger = logger
        self.target_logger = target_logger
        self.approximation_logger = approximation_logger
        self.train_logger = train_logger
        
    def initialize_for_new_task(self, query):
        """Initialize the planner for a new task."""
        # Reset steps for new task
        self.steps = []
        
        # Reset mismatch state
        self.mismatch_state.mismatch_detected.clear()
        self.mismatch_state.mismatch_step_id = None
        
        # Set query for both agents
        self.app_agent.query = query
        self.tar_agent.query = query
        
        # Reset task-specific configs using config's reset method
        self.config.reset_task_metrics()
        
    def log_task_description(self, task_description):
        """Log task description to all loggers."""
        self.logger.log('task description: ' + task_description)
        self.target_logger.log('task description: ' + task_description)
        self.approximation_logger.log('task description: ' + task_description)

    def log_results(self, steps):
        """Log final results."""
        self.logger.log('final result for the speculative planning ' + str([s[0] for s in steps]))
        self.logger.log('max concurrent calls: ' + str(self.config.MAX_CONCURRENT_CALLS-1))

    async def initialize(self, args):
        """Initialize the planner."""
        await self.mismatch_state.initialize()

    async def process_target(self, sas, tas, to_print_id, prev_steps, target_tasks):
        """Process target agent responses and update collectors."""
        s = sas[to_print_id]
        t = tas[to_print_id]
        self.target_logger.log(f'Target: Step {t[0][0] + len(prev_steps)+1} - {t[0][1][0]}')
        
        # Add to online trajectory collector
        cur_time = time.time()
        timestamp = datetime.fromtimestamp(cur_time)
        source = "Target"
        step = t[0][0] + len(prev_steps) + 1
        desc = t[0][1][0]
        self.collector.record_step(timestamp, source, step, desc)
        
        self.config.TOTAL_APPROXIMATION_CALLS += 1
        if judge_to_be_true(s[0], t[0][1][0]):
            self.config.TOTAL_CORRECT_APPROXIMATION_CALLS += 1
            self.logger.log(f'The target agent thinks step {len(prev_steps) + to_print_id+1} should be {t[0][1][0]}, which agrees with the approximation agent.')
            try:
                self.logger.log(f'The approximation agent thinks step {len(prev_steps) + to_print_id+2} should be {sas[to_print_id+1][0]}')
                register_async_handler(target_tasks=target_tasks)
            except Exception:
                pass
        else:
            self.logger.log(f'The target agent thinks step {len(prev_steps) + to_print_id+1} should be {t[0][1][0]}, correcting what the approximation agent thinks which is {s[0]}.')

    def process_on_time_interactions(self, sas, tas, flatten_tas, printed_ids, prev_steps, target_tasks):
        """Handle on-time interaction between approximation and target agents."""
        tas_ids = [t[0] for t in flatten_tas]
        flatten_printed_ids = [i for ids in printed_ids if ids for i in ids]
        tas_length = len(tas_ids)
        
        for l in range(tas_length+1):
            if not(tas_ids[:l] == list(range(len(flatten_tas)))[:l] and len(sas) >= len(flatten_tas[:l])):
                tas_length = l-1
                break

        contain_wrong_result = any(
            not judge_to_be_true(sas[pid][0], tas[pid][0][1][0]) for pid in flatten_printed_ids
        )

        if tas_length > 0 and not contain_wrong_result:
            tas_ids = tas_ids[:tas_length]
            to_print_ids = list(set(tas_ids) - set(flatten_printed_ids))
            for order_id, to_print_id in enumerate(to_print_ids):
                if order_id > 0 and not judge_to_be_true(
                        sas[to_print_ids[order_id-1]][0], 
                        tas[to_print_ids[order_id-1]][0][1][0]
                    ):
                        break
                printed_ids[to_print_id].append(to_print_id)
                t_res = interaction_function(sas[to_print_id], tas[to_print_id], self.logger, self.collector, prev_steps, self.config)
                if str(t_res) == str(tas[to_print_id][0][1][0]):
                    if t_res.lower() == "terminate":
                        return printed_ids, tas
                    continue

                changed_position = list(tas[to_print_id][0])
                self.tar_agent.execute(t_res)
                changed_position[1] = [
                    t_res,
                    self.tar_agent.current_observation,
                ]
                tas[to_print_id][0] = changed_position
                return printed_ids, tas

        return printed_ids, tas

    def process_remaining_interactions(self, sas, tas, flatten_tas, printed_ids, prev_steps, target_tasks):
        """Handle remaining interactions and process corrections."""
        flatten_printed_ids = [i for ids in printed_ids if ids for i in ids]
        
        print_leftover = (
            len(sas) > len(flatten_printed_ids)
            and len(flatten_tas) > len(flatten_printed_ids)
            and all(judge_to_be_true(sas[pid][0], tas[pid][0][1][0]) for pid in flatten_printed_ids)
        )
        
        contain_wrong_result = any(
            not judge_to_be_true(sas[pid][0], tas[pid][0][1][0]) for pid in flatten_printed_ids
        )

        if print_leftover and not contain_wrong_result:
            for print_id in range(len(flatten_printed_ids), min(len(sas),len(tas))):
                if not tas[print_id]:
                    break

                t_res = interaction_function(sas[print_id], tas[print_id], self.logger, self.collector, prev_steps, self.config)
                printed_ids[print_id].append(print_id)
                if str(t_res) == str(tas[print_id][0][1][0]):
                    continue

                changed_position = list(tas[print_id][0])
                self.tar_agent.execute(t_res)
                changed_position[1] = [
                    t_res,
                    self.tar_agent.current_observation,
                ]
                tas[print_id][0] = changed_position
            
                if not judge_to_be_true(sas[print_id][0], tas[print_id][0][1][0]):
                    break     

        return printed_ids, tas


    async def approx_gen(self, total_step_number, start):
        """Generate approximation agent responses."""
        # Get the response from the agent
        action, finished = await self.app_agent.direct_act(total_step_number, self.config, self.approximation_logger)
        
        # Add approximation task to online trajectory collector
        end = time.time()
        timestamp = datetime.fromtimestamp(end)
        source = "Approximation"
        step = total_step_number+1
        desc = action
        self.collector.record_step(timestamp, source, step, desc)
        
        # Record timing
        a_time = round(end-start, 2)

        app_tokens = self.config.APPROX_NORMAL_GENERATION[total_step_number]
        self.approximation_logger.log(
            f'Approximation: Step {total_step_number+1} -Action {action} -Finished {finished} -time {str(a_time)} -token {app_tokens}'
        )
        self.config.APPROX_NROMAL_TIME[total_step_number+1] = a_time
        
        # find action
        if not finished:
            self.app_agent.execute(action)
            observation = self.app_agent.current_observation
        else:
            observation = "terminate"

        return action, observation

    async def target_gen(self, args, prediction_task, prompt, total_step_number, tas, sas, target_tasks, printed_ids, current_step, prev_steps, start):
        """Generate target agent responses with comprehensive error handling."""
        result = None
        finished = False
        
        n = 0
        while True:
            try:
                n += 1
                if n >= 10:
                    result = ''
                    break
                
                # Set the query for the agent
                self.tar_agent.query = prompt
                
                # Create scratchpad
                scratchpad = ""
                scratchpad = self.tar_agent.create_scratchpad(scratchpad, prev_steps + sas)
                
                # Get response from agent
                action, finished = await self.tar_agent.think_and_act(
                    scratchpad, total_step_number, self.config, self.target_logger
                )
                result = action
                break
            except Exception:
                await asyncio.sleep(0.1)
                continue
        
        tas, printed_ids = await self.verify_process(
            args, prediction_task, result, finished,
            total_step_number, tas, sas, target_tasks, 
            printed_ids=printed_ids, current_step=current_step,
            prev_steps=prev_steps,
            start=start
        )

        # Record timing
        end = time.time()
        t_time = round(end-start, 2)
        self.config.TARGET_NORMAL_TIME[total_step_number+1] = t_time

        self.target_logger.log(
            f"Intermediate Target Step {total_step_number+1} -Action {result} -Finished {finished} -gen {self.config.TARGET_NORMAL_GENERATION[total_step_number+1]} -prompt {self.config.TARGET_NORMAL_PROMPT[total_step_number+1]} -time {t_time}"
        )

        return tas, printed_ids

    async def verify_process(self, args, prediction_task, result, finished, total_step_number, tas, sas, target_tasks, printed_ids=[[]], current_step=0, prev_steps=[], start=0):
        """Post-process target generation results and handle mismatches."""
        in_step_number = total_step_number - current_step
        tas[in_step_number] = [[in_step_number, (result, finished)]]

        flatten_tas = []
        for t in tas:
            if t:
                flatten_tas += t
        flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
        printed_ids, tas = self.process_on_time_interactions(
            sas, tas, flatten_tas, printed_ids,
            prev_steps, target_tasks
        )

        flatten_ids = [t[0] for t in flatten_tas]
        if flatten_ids == list(range(len(flatten_ids))):
            for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
                self.target_logger.log(f"step number:{step_number}, t[0]: {t[0]}, t[1]: {t[1]}")
                if t[0] == step_number and t[1][1]:  # terminate
                    if not self.mismatch_state.mismatch_detected.is_set():
                        self.mismatch_state.mismatch_step_id = t[0]
                        self.mismatch_state.mismatch_detected.set()
                    end = time.time()
                    t_time = round(end-start, 2)
                    self.config.TARGET_NORMAL_TIME[total_step_number+1] = t_time

                    self.target_logger.log(
                        f"Intermediate Target Step {total_step_number+1} -Action {result} -Finished {finished} -gen {self.config.TARGET_NORMAL_GENERATION[total_step_number+1]} -prompt {self.config.TARGET_NORMAL_PROMPT[total_step_number+1]} -time {t_time}"
                    )

                    raise Exception('terminate the whole process!')

        flatten_tas = []
        for t in tas:
            if t:
                flatten_tas += t
        flatten_tas = sorted(flatten_tas, key=lambda x: x[0],reverse=False)
        for ta in flatten_tas:
            if len(sas) > ta[0]:
                if not judge_to_be_true(sas[ta[0]][0], ta[1][0]) or ta[1][1] or ta[1][0].lower() == "terminate":
                    if not self.mismatch_state.mismatch_detected.is_set():
                        self.mismatch_state.mismatch_step_id = ta[0]
                        self.mismatch_state.mismatch_detected.set()
                    
                    pending_approximation_tasks = [
                        t for t in asyncio.all_tasks() 
                        if not t.cancelled() and not t.done() 
                        and t not in target_tasks 
                        and t.get_name().startswith('approximation')
                    ]
                    
                    for pending_approximation_task in pending_approximation_tasks:
                        await cancel(pending_approximation_task)
                    
                    end = time.time()
                    t_time = round(end-start, 2)
                    self.config.TARGET_NORMAL_TIME[total_step_number+1] = t_time
                    self.target_logger.log(         
                        f"Intermediate Target Step {total_step_number+1} -Action {result} -Finished {finished} -gen {self.config.TARGET_NORMAL_GENERATION[total_step_number+1]} -prompt {self.config.TARGET_NORMAL_PROMPT[total_step_number+1]} -time {t_time}"
                    )
                    raise Exception(f'approximation error happen in step {total_step_number} for current step {current_step}, the target id is {ta[0]}')
        
        if self.config.ENABLE_PRED:
            k = await prediction_task
        else:
            k = args.k

        if not self.mismatch_state.mismatch_detected.is_set() and ta[0] >= k-1:
            self.mismatch_state.mismatch_step_id = ta[0]
            self.mismatch_state.mismatch_detected.set()

        return tas, printed_ids

    async def one_episode_sp(self, args, app_prompt, tar_prompt, current_step):
        """Run one breaking point of speculative planning."""
        sas = [] # approximation
        tas = [] # target
        target_tasks = []
        printed_ids = []
        
        self.mismatch_state.mismatch_detected.clear()
        pred_k = 1 if self.config.ENABLE_PRED else args.k
        first = True
        
        i = 0
        prediction_task = None
        while i < pred_k:
            if self.mismatch_state.mismatch_detected.is_set():
                break
                
            if self.config.ENABLE_PRED and first:
                prediction_task = asyncio.create_task(
                    self.executor.async_predict(self.approximation_logger, args.offset)
                )
                self.config.PENDING_BACKGROUND_TASKS.append(prediction_task)
            
            break_out_approximation = False
            tas.append([])
            printed_ids.append([])
            
            # Generate approximation
            a_start = time.time()
            approximation = asyncio.create_task(
                self.approx_gen(
                    current_step+i, a_start
                ),
                name=f"approximation_{current_step+i}"
            )
            
            # Generate target
            t_start = time.time()
            target = asyncio.create_task(
                self.target_gen(
                    args, prediction_task, tar_prompt, current_step+i, tas, sas,
                    target_tasks, printed_ids, current_step,
                    self.steps, t_start
                ),
                name=f"target_{current_step+i}"
            )
            target_tasks.append(target)

            # Track concurrent calls
            concurrent_api_calls = concurrent_calls()
            if concurrent_api_calls >= self.config.MAX_CONCURRENT_CALLS:
                self.config.MAX_CONCURRENT_CALLS = concurrent_api_calls

            try:
                action, observation = await approximation
                sa = [action, observation]
                sas.append(sa)
                if self.mismatch_state.mismatch_detected.is_set():
                    break
            except asyncio.CancelledError:
                pass

            # Check for termination conditions
            if observation is True or action.lower == 'terminate':  # terminate
                break_out_approximation = True

            flatten_tas = []
            for t in tas:
                if t:
                    flatten_tas += t
            flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
            
            for t in flatten_tas:
                if len(sas) > t[0]:
                    if not judge_to_be_true(sas[t[0]][0], t[1][0]) or t[1][1] or t[1][0].lower() == "terminate":
                        break_out_approximation = True
                        self.mismatch_state.mismatch_step_id = t[0]
                        self.mismatch_state.mismatch_detected.set()

                        for process_id, one_task in enumerate(target_tasks):
                            if not one_task.cancelled() and not one_task.done() and process_id > t[0]:
                                self.target_logger.log(f'Cancel Task {len(self.steps)+process_id+1}')
                                await cancel(one_task)
                        break

            if break_out_approximation:
                break
            
            if self.config.ENABLE_PRED and first:
                pred_k = await prediction_task
                self.config.PREDICT_K.append(pred_k)
                self.config.PREDICT_TOTAL += 1
                pred_k = max(pred_k, 0)
                first = False
            i += 1

        # Process results after breaking point
        await self.mismatch_state.mismatch_detected.wait()
        self.target_logger.log(f"Breaking point stops at step {len(self.steps)+1+self.mismatch_state.mismatch_step_id}. Start Cancellation.")
        
        for process_id, one_task in enumerate(target_tasks):
            if not one_task.cancelled() and not one_task.done() and process_id > self.mismatch_state.mismatch_step_id:
                self.target_logger.log(f'Cancel task {len(self.steps)+process_id+1}')
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
                    except Exception:
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
                        if not one_task.cancelled() and not one_task.done() and process_id > self.mismatch_state.mismatch_step_id:
                            self.target_logger.log(f"cancel task at line 552 {len(self.steps)+process_id+1}")
                            await cancel(one_task)
                    # organize the results and return the final results
                    flatten_tas = []
                    for t in tas:
                        if t:
                            flatten_tas += t
                    flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
                    printed_ids, tas = self.process_remaining_interactions(
                        sas, tas, flatten_tas, printed_ids,
                        self.steps, target_tasks
                    )

                    # get the final tas result
                    flatten_tas = []
                    for t in tas:
                        if t:
                            flatten_tas += t
                    flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
                    for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
                        if t[0] == step_number and not judge_to_be_true(s[0], t[1][0]):
                            self.tar_agent.execute(t[1][0])
                            to_replace_action = [
                                flatten_tas[step_number][1][0],
                                self.tar_agent.current_observation,
                            ]
                            sas = sas[:step_number] + [to_replace_action]
                            self.app_agent.update_scratchpad(
                                sas[-1][0], sas[-1][1], len(self.steps) + step_number
                            )
                            self.config.PREDICT_CORRECT += self.collector.build_trajectory(self.target_logger, self.config.PREDICT_K)
                            self.config.PREDICT_K = [] 
                            if self.config.ENABLE_TRAIN:
                                if self.config.BUILD_TRAJ_TIMES == 0:
                                    train_task = asyncio.create_task(self.executor.async_train(self.train_logger))
                                    self.config.PENDING_BACKGROUND_TASKS.append(train_task)
                                self.config.BUILD_TRAJ_TIMES = (self.config.BUILD_TRAJ_TIMES + 1) % self.config.TRAIN_INTERVAL
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
                        except Exception:
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
                            self.target_logger.log(f"cancel task at line 608 {len(self.steps)+process_id+1}")
                            await cancel(one_task)

                    pending_tasks = [
                        t for process_id, t in enumerate(target_tasks)
                        if not t.cancelled() and process_id != mistaken_process_id
                    ]

            if break_while_loop:
                break

        # Process leftover interactions
        flatten_tas = sorted([item for t in tas if t for item in t], key=lambda x: x[0], reverse=False)
        printed_ids, tas = self.process_remaining_interactions(
            sas, tas, flatten_tas, printed_ids,
            self.steps, target_tasks
        )

        flatten_tas = sorted([item for t in tas if t for item in t], key=lambda x: x[0], reverse=False)
        
        # self.target_logger.log(f"sas tasks: {[(sa[0], False if sa[1] is not True else True) for sa in sas]}")    
        # self.target_logger.log(f"flatten_tas: {flatten_tas}")
     
        all_match = True
        for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
            if (len(self.steps)+step_number+1 >= self.config.MAX_STEP) or (t[0] == step_number and (not judge_to_be_true(s[0], t[1][0]) or (t[1][0].lower() == "terminate" or t[1][1]==True))):  # t[1] != s:
                all_match = False
                self.config.PREDICT_CORRECT += self.collector.build_trajectory(self.target_logger, self.config.PREDICT_K)
                self.config.PREDICT_K = [] 
                
                if self.config.ENABLE_TRAIN:
                    if self.config.BUILD_TRAJ_TIMES == 0:
                        train_task = asyncio.create_task(self.executor.async_train(self.train_logger))
                        self.config.PENDING_BACKGROUND_TASKS.append(train_task)
                    self.config.BUILD_TRAJ_TIMES = (self.config.BUILD_TRAJ_TIMES + 1) % self.config.TRAIN_INTERVAL
                
                t1 = time.time()
                self.tar_agent.execute(t[1][0])
                t2 = time.time()
                self.config.TARGET_NORMAL_TIME[len(self.steps)+step_number+1] += round(t2-t1, 2)
                to_replace_action = [
                    flatten_tas[step_number][1][0],
                    self.tar_agent.current_observation,
                ]
                sas = sas[:step_number] + [to_replace_action]
                self.app_agent.update_scratchpad(
                    sas[-1][0], sas[-1][1], len(self.steps) + step_number
                )

                break
        
        if all_match:
            sas = sas[:len(flatten_tas)]
        t2 = time.time()    
        return sas 

    async def run(self, args, prompt: str) -> List[str]:
        """Run the speculative planning process."""
        # Initialize agents and state
        await self.initialize(args)
        
        begin_time = datetime.now()
        steps = []
        breaking_points = 0
        i = 0

        app_prompt = prompt
        tar_prompt = prompt
        
        while True:
            result = await self.one_episode_sp(
                args, app_prompt, tar_prompt, len(steps)
            )
            
            steps += result
            self.steps = steps  # Update self.steps to match original behavior
            breaking_points += 1
            i += len(result)

            # if the last action is terminate, we break the generation process
            if result[-1][0].lower() == "terminate" or result[-1][1] is True or len(steps) >= self.config.MAX_STEP:
                break

        end_time = datetime.now()
        self.logger.log(f"{end_time} - {begin_time} = {end_time - begin_time}")
        self.config.TOTAL_SP_TIME = round((end_time - begin_time).total_seconds(), 2)
        
        return steps 
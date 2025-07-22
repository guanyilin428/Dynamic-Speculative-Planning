"""Dynamic Speculative Planning implementation."""

import asyncio
import time
from datetime import datetime
from typing import List

from .openagi_utils import judge_to_be_true, concurrent_calls
from util import cancel, register_async_handler
from .config import Config
from .async_online_utils import SharedState


class SpeculativePlanner:
    def __init__(self, collector, encoding, executor, app_agent, tar_agent, config: Config):
        self.collector = collector
        self.encoding = encoding
        self.executor = executor
        self.app_agent = app_agent
        self.tar_agent = tar_agent
        self.config = config
        self.mismatch_state = SharedState()
        self.steps = []
        
        # Initialize loggers as None - will be set per task
        self.logger = None
        self.target_logger = None
        self.approximation_logger = None

    def set_loggers(self, logger, target_logger, approximation_logger):
        self.logger = logger
        self.target_logger = target_logger
        self.approximation_logger = approximation_logger
        
        # Update agent loggers
        self.app_agent.logger = approximation_logger
        self.tar_agent.logger = target_logger

    def log_task_description(self, task_description):
        self.logger.log('task description: ' + task_description)
        self.target_logger.log('task description: ' + task_description)
        self.approximation_logger.log('task description: ' + task_description)

    def log_results(self, steps):
        self.logger.log('final result for the speculative planning ' + str(steps))
        self.logger.log('max concurrent calls: ' + str(self.config.MAX_CONCURRENT_CALLS-1))

    async def initialize(self, args):
        await self.mismatch_state.initialize()

    async def process_target(self, sas, tas, to_print_id, prev_steps, target_tasks):
        """Process target agent responses and update collectors."""
        s = sas[to_print_id]
        t = tas[to_print_id]
        self.target_logger.log(f'Target: Step {t[0][0] + len(prev_steps)+1} - {t[0][1]}')
        
        # Add to online trajectory collector
        cur_time = time.time()
        timestamp = datetime.fromtimestamp(cur_time)
        source = "Target"
        step = t[0][0] + len(prev_steps) + 1
        desc = t[0][1]
        self.collector.record_step(timestamp, source, step, desc)
        
        self.config.TOTAL_APPROXIMATION_CALLS += 1
        if judge_to_be_true(s, t[0][1]):
            self.config.TOTAL_CORRECT_APPROXIMATION_CALLS += 1
            self.logger.log(f'The target agent thinks step {len(prev_steps) + to_print_id+1} should be {t[0][1]}, which agrees with the approximation agent.')
            try:
                self.logger.log(f'The approximation agent thinks step {len(prev_steps) + to_print_id+2} should be {sas[to_print_id+1]}')
                self.config.HIL_INTERACTION = to_print_id+1
                register_async_handler(target_tasks=target_tasks)
            except Exception:
                pass
        else:
            self.logger.log(f'The target agent thinks step {len(prev_steps) + to_print_id+1} should be {t[0][1]}, correcting what the approximation agent thinks which is {s}.')

    async def process_on_time_interactions(self, sas, tas, flatten_tas, printed_ids, prev_steps, target_tasks):
        """Handle on-time interaction between approximation and target agents."""
        tas_ids = [t[0] for t in flatten_tas]
        flatten_printed_ids = [i for ids in printed_ids if ids for i in ids]

        tas_length = len(tas_ids)
        
        for l in range(0, tas_length+1):
            if not(tas_ids[:l] == list(range(len(flatten_tas)))[:l] and len(sas) >= len(flatten_tas[:l])):
                tas_length = l-1
                break

        contain_wrong_result = any(
            not judge_to_be_true(sas[pid], tas[pid][0][1]) for pid in flatten_printed_ids
        )

        if tas_length > 0 and not contain_wrong_result:
            tas_ids = tas_ids[:tas_length]
            to_print_ids = list(set(tas_ids) - set(flatten_printed_ids))
            for order_id, to_print_id in enumerate(to_print_ids):
                if order_id > 0 and not judge_to_be_true(
                    sas[to_print_ids[order_id-1]], 
                    tas[to_print_ids[order_id-1]][0][1]
                ):
                    break
                printed_ids[to_print_id].append(to_print_id)
                await self.process_target(sas, tas, to_print_id, prev_steps, target_tasks)

        return printed_ids, tas

    async def process_remaining_interactions(self, sas, tas, flatten_tas, printed_ids, prev_steps, target_tasks):
        """Handle remaining interactions and process corrections."""
        flatten_printed_ids = [i for ids in printed_ids if ids for i in ids]

        
        print_leftover = (
            len(sas) > len(flatten_printed_ids)
            and len(flatten_tas) > len(flatten_printed_ids)
            and all(judge_to_be_true(sas[pid], tas[pid][0][1]) for pid in flatten_printed_ids)
        )

        contain_wrong_result = any(
            not judge_to_be_true(sas[pid], tas[pid][0][1]) for pid in flatten_printed_ids
        )

        if print_leftover and not contain_wrong_result:
            for print_id in range(len(flatten_printed_ids), min(len(sas),len(tas))):
                if not tas[print_id]:
                    break
                
                await self.process_target(sas, tas, print_id, prev_steps, target_tasks)
                if not judge_to_be_true(sas[print_id], tas[print_id][0][1]):
                    break

        return printed_ids, tas

    async def A_generate(self, args, collector, app_prompt, total_step_number, start):
        """Generate approximation agent responses."""
        try:
            # Get the response from the agent (token tracking handled by agent)
            response = await self.app_agent.generate(app_prompt, total_step_number + 1)
            
            # Add approximation task to online trajectory collector
            end = time.time()
            timestamp = datetime.fromtimestamp(end)
            source = "Approximation"
            step = total_step_number+1
            desc = response
            collector.record_step(timestamp, source, step, desc)
            
            # Record timing
            a_time = round(end-start, 2)
            app_tokens = self.config.APPROX_NORMAL_GENERATION[total_step_number+1]
            self.approximation_logger.log(
                f'Approximation: Step {total_step_number+1} - {response} -time {str(a_time)} -token {app_tokens}'
            )
            self.config.APPROX_NORMAL_TIME[total_step_number+1] = a_time
            
            return response
        except Exception:
            return ''

    async def T_generate(self, args, prediction_task, prompt, total_step_number, tas, sas, target_tasks, printed_ids, current_step, prev_steps, start):
        """Generate target agent responses with comprehensive error handling."""
        result = None
        
        try:
            n = 0
            while True:
                try:
                    n += 1
                    if n >= 10:
                        result = ''
                        break
                    
                    # Get response from agent (token tracking handled by agent)
                    response = await self.tar_agent.generate(prompt, total_step_number + 1)
                    result = response
                    break
                except Exception:
                    await asyncio.sleep(0.1)
                    continue
            
            tas, printed_ids = await self.postprocess_T_generation(
                args, prediction_task, result, 
                total_step_number, tas, sas, target_tasks, 
                printed_ids=printed_ids, current_step=current_step,
                prev_steps=prev_steps
            )
        
        except Exception:
            pass

        # Record timing
        end = time.time()
        t_time = round(end-start, 2)
        self.config.TARGET_NORMAL_TIME[total_step_number+1] = t_time

        self.target_logger.log(
            f"Intermediate Target Step {total_step_number+1} - {result} "
            f"-gen {self.config.TARGET_NORMAL_GENERATION[total_step_number+1]} "
            f"-prompt {self.config.TARGET_NORMAL_PROMPT[total_step_number+1]} "
            f"-time {t_time}"
        )

        return tas, printed_ids

    async def postprocess_T_generation(self, args, prediction_task, result, total_step_number, tas, sas, target_tasks, printed_ids=[[]], current_step=0, prev_steps=[]):
        """Post-process target generation results and handle mismatches."""
        in_step_number = total_step_number - current_step
        tas[in_step_number].append((in_step_number,result))

        flatten_tas = sorted([item for t in tas if t for item in t], key=lambda x: x[0], reverse=False)
        printed_ids, tas = await self.process_on_time_interactions(
            sas, tas, flatten_tas, printed_ids,
            prev_steps, target_tasks
        )

        flatten_ids = [t[0] for t in flatten_tas]
        if flatten_ids == list(range(len(flatten_ids))):
            for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
                self.target_logger.log(f"step number:{step_number}, t[0]: {t[0]}, t[1]: {t[1]}")
                if t[1].lower() == 'terminate':
                    if not self.mismatch_state.mismatch_detected.is_set():
                        self.mismatch_state.mismatch_step_id = t[0]
                        self.mismatch_state.mismatch_detected.set()
                    raise Exception('terminate the whole process!')

        flatten_tas = sorted([item for t in tas if t for item in t], key=lambda x: x[0], reverse=False)
        for ta in flatten_tas:
            if len(sas) > ta[0]:
                if not judge_to_be_true(sas[ta[0]], ta[1]) or ta[1].lower() == "terminate":
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
            
            break_out_approximation = False
            tas.append([])
            printed_ids.append([])
            
            # Generate approximation
            a_start = time.time()
            approximation = asyncio.create_task(
                self.A_generate(
                    args, self.collector, app_prompt, current_step+i, a_start
                ),
                name=f"approximation_{current_step+i}"
            )
            
            # Generate target
            t_start = time.time()
            target = asyncio.create_task(
                self.T_generate(
                    args, prediction_task, tar_prompt, current_step+i, tas, sas,
                    target_tasks, printed_ids, current_step,
                    self.steps, t_start
                ),
                name=f"target_{i}"
            )
            target_tasks.append(target)

            # Track concurrent calls
            concurrent_api_calls = concurrent_calls()
            if concurrent_api_calls >= self.config.MAX_CONCURRENT_CALLS:
                self.config.MAX_CONCURRENT_CALLS = concurrent_api_calls

            try:
                sa = await approximation
                sas.append(sa)
                if self.mismatch_state.mismatch_detected.is_set():
                    break

                # Check if we need to print approximation result
                flatten_tas = []
                for t in tas[:len(sas)-1]:
                    if t:
                        flatten_tas += t
                flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
                flatten_ids = [t[0] for t in flatten_tas]
                flatten_tas_action = [t[1] for t in flatten_tas]
                if (flatten_ids == list(range(len(flatten_ids))) and 
                    len(flatten_ids) == len(sas)-1 and 
                    all([judge_to_be_true(s, t) for s, t in zip(sas[:-1], flatten_tas_action)])):
                    
                    flattened_printed_ids = [printed_id[0] for printed_id in printed_ids if printed_id != []]
                    
                    if flattened_printed_ids:
                        if len(sas) == len(flattened_printed_ids)+1:
                            self.logger.log(f'The approximation agent thinks step {current_step+i+1} should be {sa}')
                            self.config.HIL_INTERACTION = len(sas)-1
                            register_async_handler(target_tasks=target_tasks)
                    else:
                        self.logger.log(f'The approximation agent thinks step {current_step+i+1} should be {sa}')
                        self.config.HIL_INTERACTION = len(sas)-1
                        register_async_handler(target_tasks=target_tasks)

                # Update prompts
                if '## Current Action Trajectory:' not in app_prompt:
                    app_prompt += '\n\n## Current Action Trajectory:\n'
                app_prompt += f'\nAction {current_step+i+1}: {str(sa)}.'
                
                if '## Current Action Trajectory:' not in tar_prompt:
                    tar_prompt += '\n\n## Current Action Trajectory:\n'
                tar_prompt += f'\nAction {current_step+i+1}: {str(sa)}.'
                
            except asyncio.CancelledError:
                self.target_logger.log("Exception happens.")
                pass

            # Check for termination conditions
            if sa.lower() == 'terminate':
                break_out_approximation = True

            flatten_tas = []
            for t in tas:
                if t:
                    flatten_tas += t
            flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
            
            for t in flatten_tas:
                if len(sas) > t[0]:
                    if not judge_to_be_true(sas[t[0]], t[1]) or t[1].lower() == "terminate":
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
                pred_k = max(pred_k, 1)
                first = False
            i += 1

        # Process results after breaking point
        await self.mismatch_state.mismatch_detected.wait()
        self.target_logger.log(f"Mismatch at {self.mismatch_state.mismatch_step_id}. Cancel starts.")
        
        for process_id, one_task in enumerate(target_tasks):
            if not one_task.cancelled() and not one_task.done() and process_id > self.mismatch_state.mismatch_step_id:
                self.target_logger.log(f'Cancel task {len(self.steps)+process_id+1}')
                await cancel(one_task)

        # Wait for pending tasks prior to mismatch task
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
                flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
                for t in flatten_tas:
                    if len(sas) > t[0]:
                        if not judge_to_be_true(sas[t[0]], t[1]):
                            for process_id, one_task in enumerate(pending_tasks):
                                if not one_task.cancelled() and not one_task.done() and process_id > t[0]:
                                    self.target_logger.log(f'Cancel task: {len(self.steps)+process_id+1}')
                                    await cancel(one_task)
                            break
            except Exception as e:
                print(f"An error occurred with task {len(self.steps)+t[0]+1}: {e}")

        # Process leftover interactions
        flatten_tas = []
        for t in tas:
            if t:
                flatten_tas += t
        flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
        printed_ids, tas = await self.process_remaining_interactions(
            sas, tas, flatten_tas, printed_ids,
            self.steps, target_tasks
        )

        # Get final results
        flatten_tas = []
        for t in tas:
            if t:
                flatten_tas += t
        flatten_tas = sorted(flatten_tas, key=lambda x: x[0], reverse=False)
        mismatch = False
        origin_sa = None
        
        self.target_logger.log(f"sas tasks: {sas}")    
        self.target_logger.log(f"flatten_tas: {flatten_tas}")

        for step_number, (s, t) in enumerate(zip(sas, flatten_tas)):
            if t[0] == step_number and (not judge_to_be_true(s, t[1]) or t[1].lower() == "terminate"):
                self.target_logger.log(f"step {step_number} app: {s}, tar: {t[1]}")
                self.config.PREDICT_CORRECT += self.collector.build_trajectory(
                    self.target_logger, self.config.PREDICT_K
                )
                self.config.PREDICT_K = []

                if self.config.ENABLE_TRAIN:
                    if self.config.BUILD_TRAJ_TIMES == 0:
                        asyncio.create_task(self.executor.async_train())
                    self.config.BUILD_TRAJ_TIMES = (
                        self.config.BUILD_TRAJ_TIMES + 1
                    ) % self.config.TRAIN_INTERVAL
                    
                sas = sas[:step_number]+[flatten_tas[step_number][1]]
                origin_sa = s
                mismatch = True
                break
        
        return sas, origin_sa, mismatch

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
            result, origin_sa, mismatch = await self.one_episode_sp(
                args, app_prompt, tar_prompt, len(steps)
            )
            
            # Update prompts based on results
            if '## Current Action Trajectory:' not in app_prompt:
                app_prompt += '\n\n## Current Action Trajectory:\n'
            previous_action_trajectory = [
                f'\nAction {len(steps) + j+1}: {result[j]}. Your prediction aligned with Target.'
                for j in range(len(result))
            ]
            
            if mismatch:
                app_prompt += ''.join(previous_action_trajectory[:-1])
                app_prompt += f"\nAction {len(steps) + len(result)}: {result[-1]}. Target predicted {result[-1]} and corrected your prediction {origin_sa}."
            else:
                app_prompt += ''.join(previous_action_trajectory)

            if '## Current Action Trajectory:' not in tar_prompt:
                tar_prompt += '\n\n## Current Action Trajectory:\n'
            previous_action_trajectory = [
                f'\nAction {len(steps) + j+1}: {result[j]}.'
                for j in range(len(result))
            ]
            tar_prompt += ''.join(previous_action_trajectory)

            steps += result
            breaking_points += 1
            i += len(result)

            if result[-1].lower() == 'terminate' or len(steps) >= self.config.MAX_STEP:
                break

        end_time = datetime.now()
        self.logger.log(f'{end_time} - {begin_time} = {end_time - begin_time}')
        self.config.TOTAL_SP_TIME = round((end_time - begin_time).total_seconds(), 2)
        
        # self.log_results(steps)
        
        return steps 
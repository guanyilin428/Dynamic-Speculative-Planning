from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from collections import deque
import torch
import random
from typing import List, Tuple
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
import asyncio
import copy
import threading
from threading import Event
import time

import hashlib

class OnlineLearningExecutor:
    def __init__(self, 
                 model: nn.Module,
                 tokenizer,
                 initial_prompt: str,
                 buffer_size=1000,
                 batch_size_steps=32,
                 max_length=512,
                 lambda_=0.9,
                 gamma=1,
                 lr=5e-5,
                 epoch_per_train=1):
        
        self.tokenizer = tokenizer
        self.buffer = FiniteReplay(tokenizer=tokenizer, model=model, max_length=max_length, replay_size=buffer_size)
        self.collector = OnlineTrajectoryCollector(initial_prompt, self.buffer)
        self.batch_size_steps = batch_size_steps
        self.criterion = nn.MSELoss()
        self.max_length = max_length
        self.lambda_ = lambda_
        self.gamma = gamma
        self.epoch_per_train = epoch_per_train
        
        self.train_model = model  
        self.predict_model = copy.deepcopy(model)  
        self.train_executor = ThreadPoolExecutor(max_workers=1)
        self.optimizer = optim.Adam(self.train_model.parameters(), lr=lr)
        self.train_lock = asyncio.Lock()
        self.model_lock = threading.Lock() 

        # self.debug_barrier = Event()  # barrier for syncing debug pause
        # self.debug_barrier.set()
        
        
    def _compute_lambda_return(self, input_ids, attention_masks, rewards):
        self.train_model.eval()
        T = len(input_ids)
        G_lambda = torch.zeros(T)
        
        G_t = 0
        for t in reversed(range(T)):
            input_id = input_ids[t].unsqueeze(0)
            attention_mask = attention_masks[t].unsqueeze(0)
            v_pred = self.train_model(input_id, attention_mask).item()   
            mask = rewards[t]
            G_t = rewards[t] + self.gamma * (1 - self.lambda_) * v_pred * mask + self.gamma * self.lambda_ * G_t * mask
            G_lambda[t] = G_t
        self.train_model.train()
        return G_lambda

    def _flat_batch_data(self, input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch):
        all_G_lambda = []
        all_input_ids = []
        all_attention_mask = []
        all_gt_k = []
        # compute current G_lambda for batch data and flatten steps
        for i in range(len(input_ids_batch)):
            # [steps_num, hid_dim]
            input_ids = input_ids_batch[i]
            attention_mask = attention_mask_batch[i]
            rewards = rewards_batch[i]

            with torch.no_grad():
                # G_lambda for each trajectory
                G_lambda = self._compute_lambda_return(input_ids, attention_mask, rewards)
            # flatten inputs and targets
            all_G_lambda.extend(G_lambda)
            all_input_ids.extend(input_ids)
            all_attention_mask.extend(attention_mask)
            all_gt_k.extend(gt_k_batch[i])
            
        device = self.train_model.model.device
        # [total_step_num, hid_dim]: total_step_num in this batch
        flat_input_ids = torch.stack(all_input_ids, dim=0).to(device)
        flat_attention_mask = torch.stack(all_attention_mask, dim=0).to(device)
        # [total_step_num]
        flat_G_lambda = torch.tensor(all_G_lambda).to(device)
        flat_gt_k = torch.stack(all_gt_k, dim=0).to(device)
        return flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k
        
    async def async_train(self, logger):
        async with self.train_lock:
            loop = asyncio.get_running_loop()
            # logger.log("Starting training")
            self.current_train_task = loop.run_in_executor(
                self.train_executor,
                self._train,
                logger
            )

    def _train(self, logger):
        for epoch in range(self.epoch_per_train):
            self.train_model.train()
            batch = self.buffer.sample(self.batch_size_steps)
        
            if batch is None:
                return
            # start = time.time()
            input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch = batch

            flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k = self._flat_batch_data(
                input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch)

            self.optimizer.zero_grad()
            k_pred = self.train_model(flat_input_ids, flat_attention_mask).squeeze()
            loss = self.criterion(k_pred, flat_G_lambda)
            diff = torch.round(k_pred) - flat_gt_k
            acc = ((diff == 0) | (diff == 1)).float().mean().item()
            logger.log(f'Epoch {epoch + 1} - Loss: {loss.item()} - Accuracy: {round(acc * 100, 2)}%')
            # self.debug_barrier.set()

            loss.backward()
            self.optimizer.step()
            
        with self.model_lock:
            self.predict_model.load_state_dict(self.train_model.state_dict())
        # logger.log(f'Trained for {self.epoch_per_train} epoch -time {round(time.time()-start, 2)}s')
    
    
    async def async_predict(self, logger):
        # logger.log("Start Prediction")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._predict,
            logger
        )

    def _predict(self, logger):
        self.predict_model.eval()
        start = time.time()
        
        state = self.collector.get_current_trajectory()
        with self.model_lock:
            # param_tensor = torch.cat([p.flatten() for p in self.predict_model.parameters()])
            # checksum = hashlib.md5(param_tensor.detach().cpu().numpy().tobytes()).hexdigest()
            # logger.log(f"Predict model checksum: {checksum}") # log model updates

            inputs = self.tokenizer(
                state,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=self.max_length
            )
            k_pred = self.predict_model(**inputs).squeeze()
        k = int(torch.round(k_pred).item())
        logger.log(f'Predictor time: {round(time.time()-start, 2)}s')
        logger.log(f'Predict K: {k}')
        return k
        

class OnlineTrajectoryCollector:
    def __init__(self, initial_task, replay_buffer: "FiniteReplay"):
        self.reset(initial_task)
        self.replay_buffer = replay_buffer
        
    def reset(self, initial_task):
        self.initial_task = initial_task        
        self.prefix = initial_task
        self.approx_logs = [] # approximation_steps in current breakingpoint
        self.target_logs = [] # target steps in current breakingpoint
    
    def _reset_trajectory(self):
        self.approx_logs = []
        self.target_logs = []
    
    def record_step(self, 
                   timestamp: datetime,
                   source: str,  # "approximation" or "target"
                   step: int,
                   description: str = None):

        if source.strip() == "Approximation":
            self.approx_logs.append((timestamp, source.strip(), step, description))
        else:
            self.target_logs.append((timestamp, source.strip(), step, description))
    
    def build_trajectory(self, logger):
        # start = time.time()
        # when reach a breakingpoint, call build_trajectory
        # print("Build Trajectory")
        combined_logs = sorted(self.approx_logs + self.target_logs, key=lambda x: (x[0], x[2], 0 if x[1] == "Approximation" else 1))
        target_logs_sorted_by_step = sorted(self.target_logs, key=lambda x: x[2])
        i, j = 0, 0
        trajectory = []
        
        while j < len(target_logs_sorted_by_step):
            target_timestamp, _, target_step, target_desc = target_logs_sorted_by_step[j]
            approx_timestamp, _, approx_step, approx_desc = self.approx_logs[i]
            if approx_desc == target_desc and j != len(target_logs_sorted_by_step) - 1:
                i += 1
                j += 1
            else: # mismatch, breakingpoint
                # iterate thru all approx tasks in this breakingpoint
                # generate traj from bp start to cur_approx_task(not included)                
                approx_index = 0
                while approx_index < len(self.approx_logs):
                    cur_approx_timestamp, _, cur_approx_step, cur_approx_desc = self.approx_logs[approx_index]
                    current_k = max(0, target_step - (cur_approx_step - 1))
                    state = []
                    
                    for log in combined_logs:
                        if log == self.approx_logs[approx_index]:
                            break
                        state.append(f"{log[1]} Step {log[2]}: {log[3]}")
                    reward = 1 if current_k > 0 else 0
                    # construct prompt
                    if state:
                        if self.prefix == self.initial_task:
                            self.prefix += "\nPrevious History:\n"
                        prompt = self.prefix + "\n".join(state) + "\n"
                    else:
                        prompt = self.prefix
                    trajectory.append({
                        "state": prompt,
                        "reward": reward,
                        "k": current_k
                    })
                    approx_index += 1
                break
        # update prefix
        bp_state = [f"{log[1]} Step {log[2]}: {log[3]}" for log in combined_logs]
        self.prefix += "\n".join(bp_state) + "\n"    
        logger.log(f"trajectory: {trajectory}")
        # add trajectory to replay buffer
        self.replay_buffer.add_trajectory(trajectory)
        self._reset_trajectory()
        # print(f'Build trajectory time: {round(time.time()-start, 2)}s') # 0.0s
        
        
                    
    def get_current_trajectory(self):
        # start = time.time()
        combined_logs = sorted(self.approx_logs + self.target_logs, key=lambda x: (x[0], x[2], 0 if x[1] == "Approximation" else 1))
        bp_state = [f"{log[1]} Step {log[2]}: {log[3]}" for log in combined_logs]
        if self.prefix == self.initial_task and bp_state:
            prefix = self.prefix + "\nPrevious History:\n"
        else: prefix = self.prefix
        prefix += "\n".join(bp_state) + "\n"    
        
        # print(f'Build predict state time: {round(time.time()-start, 2)}s') # 0.0s
        return prefix    
            
    def save_trajectory(self, file_path):
        with open(file_path, 'w') as f:
            json.dump({
                "task_description": self.task_description,
                "trajectory": self.current_trajectory
            }, f, indent=2)
            

class FiniteReplay:
    def __init__(self, tokenizer, model, max_length=512, replay_size: int = 100):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.replay_size = replay_size
        
        self.replay_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = [None] * replay_size
        self.trajectory_lengths: List[int] = [0] * replay_size
        self.pos = 0 
        self.full = False

    def __len__(self) -> int:
        return self.replay_size if self.full else self.pos

    def add_trajectory(self, trajectory: List[dict]):
        # breakpoint()
        processed = self._preprocess_trajectory(trajectory)
        self.replay_buffer[self.pos] = processed
        self.trajectory_lengths[self.pos] = len(trajectory)
        self.pos = (self.pos + 1) % self.replay_size
        if self.pos == 0:
            self.full = True

    def _preprocess_trajectory(self, trajectory: List[dict]):
        states = [step["state"] for step in trajectory]
        rewards = [step["reward"] for step in trajectory]
        gt_ks = [step["k"] for step in trajectory]

        inputs = self.tokenizer(
            states,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        input_ids = inputs['input_ids']
        attention_masks = inputs['attention_mask']
        rewards = torch.tensor(rewards, dtype=torch.float32)
        gt_ks = torch.tensor(gt_ks, dtype=torch.float32)
        
        return (input_ids, attention_masks, rewards, gt_ks)

    def sample(self, batch_size_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = []
        total_steps = 0
        available_indices = list(range(len(self)))
        random.shuffle(available_indices)

        input_ids_list = []
        attention_mask_list = []
        rewards_list = []
        gt_k_list = []
        total_steps = 0
        for idx in available_indices:
            input_ids, attention_mask, rewards, gt_ks = self.replay_buffer[idx]
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            rewards_list.append(rewards)
            gt_k_list.append(gt_ks)
            total_steps += self.trajectory_lengths[idx]
            if total_steps >= batch_size_steps:
                break
        
        if total_steps < batch_size_steps:
            return None
        return input_ids_list, attention_mask_list, rewards_list, gt_k_list

class SharedState:
    mismatch_detected = asyncio.Event()
    mismatch_step_id = None
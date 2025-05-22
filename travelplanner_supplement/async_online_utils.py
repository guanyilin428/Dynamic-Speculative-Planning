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
import json
import hashlib
import os
import nltk

class ExpectileLoss(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau
    
    def forward(self, G_t, pred_k):
        diff = G_t - pred_k
        weight = torch.where(diff > 0, self.tau, (1 - self.tau))
        return torch.mean(weight * diff ** 2)

class OnlineLearningExecutor:
    def __init__(self, 
                 device,
                 model_save_path,
                 model: nn.Module,
                 tokenizer,
                 buffer_size=200,
                 batch_size_steps=32,
                 max_length=512,
                 lambda_=0.9,
                 gamma=1,
                 lr=5e-5,
                 epoch_per_train=5,
                 load=False,
                 traj_file=None,
                 tau = 0.5
                 ):
        self.device = device
        self.tokenizer = tokenizer
        self.buffer = FiniteReplay(tokenizer=tokenizer, model=model, max_length=max_length, replay_size=buffer_size)
        self.collector = OnlineTrajectoryCollector(self.buffer)
        self.batch_size_steps = batch_size_steps
        if tau == 0.5:
            self.criterion = nn.MSELoss()
        else: self.criterion = ExpectileLoss(tau)
        self.max_length = max_length
        self.lambda_ = lambda_
        self.gamma = gamma
        self.epoch_per_train = epoch_per_train
        
        self.train_model = model.to(device)  
        self.predict_model = copy.deepcopy(model).to(device)  
        self.train_executor = ThreadPoolExecutor(max_workers=1)
        self.optimizer = optim.AdamW(self.train_model.parameters(), lr=lr)
        self.train_lock = asyncio.Lock()
        self.model_lock = threading.Lock() 
        self.model_save_path = model_save_path
        if load:
            self.load_checkpoint(model_save_path)
            self.load_from_file(traj_file)
    
    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.train_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.predict_model.load_state_dict(self.train_model.state_dict())
        print("load weights from last breakingpoint")

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
                G_lambda = self._compute_lambda_return(input_ids.to(self.device), attention_mask.to(self.device), rewards.to(self.device))
            # flatten inputs and targets
            all_G_lambda.extend(G_lambda)
            all_input_ids.extend(input_ids)
            all_attention_mask.extend(attention_mask)
            all_gt_k.extend(gt_k_batch[i])
            
        
        # [total_step_num, hid_dim]: total_step_num in this batch
        flat_input_ids = torch.stack(all_input_ids, dim=0).to(self.device)
        flat_attention_mask = torch.stack(all_attention_mask, dim=0).to(self.device)
        # [total_step_num]
        flat_G_lambda = torch.tensor(all_G_lambda).to(self.device)
        flat_gt_k = torch.stack(all_gt_k, dim=0).to(self.device)
        return flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k
        
    async def async_train(self, logger):
        async with self.train_lock:
            loop = asyncio.get_running_loop()
            # logger.log("Starting training")
            current_train_task = loop.run_in_executor(
                self.train_executor,
                self._train,
                logger
            )
            try:
                await current_train_task
            except Exception as e:
                logger.log(f"Training failed with exception: {e}")

    def _train(self, logger):
        batch = self.buffer.sample(self.batch_size_steps, logger)
        for epoch in range(self.epoch_per_train):
            self.train_model.train()
        
            if batch is None:
                logger.log("Batch is None, skipping training.")
                return
            input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch = batch

            flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k = self._flat_batch_data(
                input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch)
            
            self.optimizer.zero_grad()
            k_pred = self.train_model(flat_input_ids, flat_attention_mask).squeeze()
            loss = self.criterion(flat_G_lambda, k_pred)
            diff = torch.round(k_pred) - flat_gt_k
            acc = ((diff == 0) | (diff == 1)).float().mean().item()
            logger.log(f'Epoch {epoch + 1} - Loss: {loss.item()} - Accuracy: {round(acc * 100, 2)}%')

            loss.backward()
            self.optimizer.step()

        with self.model_lock:
            self.predict_model.load_state_dict(self.train_model.state_dict())
            checkpoint = {
                "model_state_dict": self.train_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, self.model_save_path)
    
    
    async def async_predict(self, logger, k_offset):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._predict,
            logger,
            k_offset
        )

    def _predict(self, logger, k_offset):
        self.predict_model.eval()
        start = time.time()
        
        state = self.collector.get_current_trajectory()
        logger.log(f"Predict state: {state}")
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
            ).to(self.device)
            k_pred = self.predict_model(**inputs).squeeze().cpu()
        k = int(torch.round(k_pred).item()) + k_offset
        # logger.log(f'Predictor time: {round(time.time()-start, 2)}s')
        logger.log(f'Predict K: {k}')
        return k
    
    def set_initial_task_prompt(self, initial_task):
        self.collector.set_initial_task_prompt(initial_task)

    def save_trajectory(self, file_path, append):
        self.collector.save_trajectory(file_path, append)

    def load_from_file(self, file_path):
        self.buffer.load_from_file(file_path)

        
class OnlineTrajectoryCollector:
    def __init__(self, replay_buffer: "FiniteReplay"):
        self.traj_list = []
        self.reset()
        self.replay_buffer = replay_buffer
        
    def reset(self):        
        self.approx_logs = [] # approximation_steps in current breakingpoint
        self.target_logs = [] # target steps in current breakingpoint
    
    def set_initial_task_prompt(self, initial_task):
        self.initial_task = initial_task        
        self.prefix = initial_task+"\n"

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
    
    
    def judge_to_be_true(self, s, t):
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

    def build_trajectory(self, logger, predict_ks):
        # start = time.time()
        # when reach a breakingpoint, call build_trajectory
        # print("Build Trajectory")
        if len(self.approx_logs) == 0:
            return 0
        # combined_logs = sorted(self.approx_logs + self.target_logs, key=lambda x: (x[0], x[2], 0 if x[1] == "Approximation" else 1))
        target_logs_sorted_by_step = sorted(self.target_logs, key=lambda x: x[2])
        i, j = 0, 0
        trajectory = []
        gt_k = 0
        # logger.log(f"\napp_logs: {self.approx_logs}\n")
        # logger.log(f"\ntar_logs: {self.target_logs}\n")
        # logger.log(f"\nprefix: {self.prefix}\n")
        # breakpoint()

        while j < len(target_logs_sorted_by_step) and i < len(self.approx_logs):
            target_timestamp, _, target_step, target_desc = target_logs_sorted_by_step[j]
            approx_timestamp, _, approx_step, approx_desc = self.approx_logs[i]
            if self.judge_to_be_true(approx_desc, target_desc) and j != len(target_logs_sorted_by_step) - 1:
                i += 1
                j += 1
            else: # mismatch, breakingpoint
                # iterate thru all approx tasks in this breakingpoint
                # generate traj from bp start to cur_approx_task(not included)   
                bp_state = [
                    f"Step {log[2]}: {log[3]}"
                    for log in self.approx_logs[:i]
                ] + [f"Step {target_step}: {target_desc}"]
                approx_index = 0
                while approx_index < len(self.approx_logs):
                    cur_approx_timestamp, _, cur_approx_step, cur_approx_desc = self.approx_logs[approx_index]
                    current_k = max(0, target_step - (cur_approx_step - 1))
                    gt_k = max(gt_k, current_k)
                    state = []
                    
                    for log in self.approx_logs[:approx_index]:
                        state.append(f"Step {log[2]}: {log[3]}")

                    reward = 1 if current_k > 0 else 0
                    # construct prompt
                    if state:
                        # if self.prefix == self.initial_task:
                        #     self.prefix += "\nPrevious History:\n"
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
        self.prefix += "\n".join(bp_state) + "\n"    
        logger.log(f"trajectory: {trajectory}")
        # add trajectory to replay buffer
        self.traj_list.append({"trajectory": trajectory})
        self.replay_buffer.add_trajectory(trajectory)
        self._reset_trajectory()

        if len(predict_ks) == 1:
            return 1 if (predict_ks[0] == gt_k or predict_ks[0] == gt_k+1) else 0
        else:
            acc = 0
            for i in range(len(predict_ks)):
                if predict_ks[i] == gt_k or predict_ks[i] == gt_k+1:
                    acc += 1
                gt_k -= predict_ks[i]
            return acc
                    
    def get_current_trajectory(self):
        bp_state = [f"Step {log[2]}: {log[3]}" for log in self.approx_logs]
        # if self.prefix == self.initial_task and bp_state:
        #     prefix = self.prefix + "\nPrevious History:\n"
        prefix = self.prefix
        prefix += "\n".join(bp_state) + "\n"    
        
        return prefix    
            
    def save_trajectory(self, file_path, append):
        if append:
            with open(file_path, 'a', encoding='utf-8') as f:
                for traj in self.traj_list:
                    f.write(json.dumps(traj) + '\n')
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.traj_list, f, indent=2)
        self.traj_list = []

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
    
    def load_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")

        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for traj_entry in data:
                        self.add_trajectory(traj_entry["trajectory"])
        elif file_path.endswith(".ndjson"):
            with open(file_path, 'r') as f:
                for line in f:
                    traj = json.loads(line)
                    self.add_trajectory(traj['trajectory'])
        else:
            raise ValueError("Unsupported file format. Use .json or .ndjson")

    def add_trajectory(self, trajectory: List[dict]):
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

    def sample(self, batch_size_steps: int, logger) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = []
        total_steps = 0
        available_indices = list(range(len(self)))
        random.shuffle(available_indices)

        input_ids_list = []
        attention_mask_list = []
        rewards_list = []
        gt_k_list = []
        total_steps = 0

        chosen_idx = []
        for idx in available_indices:
            input_ids, attention_mask, rewards, gt_ks = self.replay_buffer[idx]
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            rewards_list.append(rewards)
            gt_k_list.append(gt_ks)
            total_steps += self.trajectory_lengths[idx]
            chosen_idx.append(idx)
            # use all datapoint to train
            if total_steps >= batch_size_steps:
                break
        if total_steps < batch_size_steps:
            return None
        logger.log(f"chosen index {chosen_idx}")
        
        return input_ids_list, attention_mask_list, rewards_list, gt_k_list

class SharedState:
    def __init__(self):
        mismatch_detected = None
        mismatch_step_id = None

    async def initialize(self):
        self.mismatch_detected = asyncio.Event()
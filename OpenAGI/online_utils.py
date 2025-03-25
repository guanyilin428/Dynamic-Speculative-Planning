from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from collections import deque
import torch
import random
from typing import List, Tuple
import torch.nn as nn
import torch.optim as optim


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
        self.model = model
        self.tokenizer = tokenizer
        self.buffer = FiniteReplay(tokenizer=tokenizer, model=model, max_length=max_length, replay_size=buffer_size)
        self.collector = OnlineTrajectoryCollector(initial_prompt, self.buffer)
        self.batch_size_steps = batch_size_steps
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.max_length = max_length
        self.lambda_ = lambda_
        self.gamma = gamma
        self.epoch_per_train = epoch_per_train
        
        
    def _compute_lambda_return(input_ids, attention_masks, rewards):
        self.model.eval()
        T = len(input_ids)
        G_lambda = torch.zeros(T)
        
        G_t = 0
        for t in reversed(range(T)):
            input_id = input_ids[t].unsqueeze(0)
            attention_mask = attention_masks[t].unsqueeze(0)
            v_pred = self.model(input_id, attention_mask).item()   
            mask = rewards[t]
            G_t = rewards[t] + self.gamma * (1 - self.lambda_) * v_pred * mask + self.gamma * self.lambda_ * G_t * mask
            G_lambda[t] = G_t
        self.model.train()
        return G_lambda

    def _flat_batch_data(input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch):
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

            # G_lambda for each trajectory
            G_lambda = self._compute_lambda_return(input_ids, attention_mask, rewards)
            # flatten inputs and targets
            all_G_lambda.extend(G_lambda)
            all_input_ids.extend(input_ids)
            all_attention_mask.extend(attention_mask)
            all_gt_k.extend(gt_k_batch[i])
            
        device = self.model.model.device
        # [total_step_num, hid_dim]: total_step_num in this batch
        flat_input_ids = torch.stack(all_input_ids, dim=0).to(device)
        flat_attention_mask = torch.stack(all_attention_mask, dim=0).to(device)
        # [total_step_num]
        flat_G_lambda = torch.tensor(all_G_lambda).to(device)
        flat_gt_k = torch.stack(all_gt_k, dim=0).to(device)
        return flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k
        
    def train(self):
        # modify to ascyn
        for epoch in range(epoch_per_train):
            self.model.train()
            input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch = self.buffer.sample(self.batch_size_steps)
            flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k = self._flat_batch_data(
                input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch)
            
            # batch = self.buffer.sample(self.batch_size_steps)
            # flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k = self._flat_batch_data(**batch)
                    
            k_pred = self.model(flat_input_ids, flat_attention_mask).squeeze()
            loss = self.criterion(k_pred, flat_G_lambda)
            diff = torch.round(k_pred) - flat_gt_k
            acc = ((diff == 0) | (diff == 1)).float().mean().item()
            
            self.optimizer.zero_grad()    
            loss.backward()
            self.optimizer.step()

    def predict(self):
        self.model.eval()
        state = self.collector.get_current_trajectory()
        print("PREDICT TRAJ:", state)
        inputs = self.tokenizer(
            state,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        k_pred = self.model(**inputs).squeeze()
        
        return int(torch.round(k_pred).item()), state
        

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
    
    def build_trajectory(self):
        # when reach a breakingpoint, call build_trajectory
        combined_logs = sorted(self.approx_logs + self.target_logs, key=lambda x: (x[0], x[2], 0 if x[1] == "Approximation" else 1))
        target_logs_sorted_by_step = sorted(self.target_logs, key=lambda x: x[2])
        i, j = 0, 0
        trajectory = []
        
        while j < len(target_logs_sorted_by_step):
            target_timestamp, _, target_step, target_desc = target_logs_sorted_by_step[j]
            approx_timestamp, _, approx_step, approx_desc = self.approx_logs[i]
            if approx_desc == target_desc :
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
            
        self._reset_trajectory()
        # update prefix
        if self.prefix == self.initial_task:
            self.prefix += "Previous History:\n"
        self.prefix += "\n".join(combined_logs) + "\n"
        
        # add trajectory to replay buffer
        self.replay_buffer.add_trajectory(trajectory)
            
    def get_current_trajectory(self):
        return self.prefix    
            
    def save_trajectory(self, file_path):
        with open(file_path, 'w') as f:
            json.dump({
                "task_description": self.task_description,
                "trajectory": self.current_trajectory
            }, f, indent=2)
            
            

class FiniteReplay:
    def __init__(self, tokenizer, model, max_length=512, replay_size: int = 1000):
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

        return input_ids_list, attention_mask_list, rewards_list, gt_k_list
                    
            
            
            
            
            
"""          
class OnlineDataset(Dataset):
    def __init__(self, tokenizer, model, max_length=512, replay_size: int = 1000):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        # self.replay_buffer = deque(maxlen=replay_size)
        self.replay_buffer = []
        self.trajectory_lengths = []
        
    def __len__(self) -> int:
        return len(self.replay_buffer)
    
    def __getitem__(self, idx):
        input_ids, attention_masks, rewards = self.replay_buffer[idx]
        return input_ids, attention_masks, rewards
    
    def add_trajectory(self, trajectory):
        self.trajectory_lengths.append(len(trajectory))
        self.replay_buffer.append(self._preprocess_trajectory(trajectory))
       
    def _preprocess_trajectory(self, trajectory):
        states = [step["state"] for step in trajectory]
        rewards = [step["reward"] for step in trajectory]
        # gt_ks = [step["k"] for step in trajectory]
        
            # rewards.append(torch.tensor(reward))
            # gt_ks.append(torch.tensor(gt_k))

        inputs = tokenizer(
            states, 
            return_tensors="pt", 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )
        input_ids = inputs['input_ids']
        attention_masks = inputs['attention_mask']
        # rewards = torch.stack(rewards, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        # gt_ks = torch.stack(gt_ks, dim=0)
        return (input_ids, attention_masks, rewards)
        
    def get_dataloader(self, 
                      batch_size: int = 32,
                      shuffle: bool = True) -> DataLoader:
        batch_sampler = DynamicBatchSampler(
            trajectory_lengths=self.trajectory_lengths,
            max_steps_per_batch=batch_size,
            shuffle=False
        )
        return DataLoader(
            self,
            batch_sampler=train_batch_sampler,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
    
    def _collate_fn(self, batch: list) -> dict:
        input_ids_batch = [trajectory_data[0] for trajectory_data in batch]
        attention_mask_batch = [trajectory_data[1] for trajectory_data in batch]
        rewards_batch = [trajectory_data[2] for trajectory_data in batch]
        return input_ids_batch, attention_mask_batch, rewards_batch
"""
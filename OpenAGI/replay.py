import torch
import random
from typing import List, Tuple

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
        
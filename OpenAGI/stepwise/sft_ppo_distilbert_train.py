import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
import random
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR

def load_dataset(json_file):
    with open(json_file, "r") as f:
        dataset = json.load(f)
    return dataset

full_dataset = load_dataset("dataset_stepwise_ma.json")
from transformers import AutoConfig

import torch


def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

class SFTPredictor(nn.Module):
    def __init__(self, bert_model, hidden_dim=768, num_actions=2):
        super().__init__()
        self.bert = bert_model
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(pooled_output)


def collate_fn(batch):
    input_texts, labels = [], []
    for sample in batch:
        input_texts.append(sample["input"])
        labels.append(int(sample["output"]))
    
    inputs = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

def train_sft(num_epochs):
    for epoch in range(num_epochs):
        sft_model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for inputs, labels in train_loader:
            logits = sft_model(**inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2%}")

        eval_sft()

def eval_sft():
    sft_model.eval()
    with torch.no_grad():
        val_correct, val_samples = 0, 0
        for inputs, labels in full_loader:
            logits = sft_model(**inputs)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_samples += labels.size(0)
        val_acc = val_correct / val_samples
    print(f"Val Acc: {val_acc:.2%}")

model_path = "../weights/distilbert_base_uncased"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
bert_model = AutoModel.from_pretrained(model_path, local_files_only=True)

sft_predictor_pth = "./OpenAGI/stepwise/predictor_weights/sft_distilbert_predictor.pth"
sft_model = SFTPredictor(bert_model, hidden_dim=768, num_actions=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(sft_model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5) 
# train_sft(num_epochs=13)
# torch.save(sft_model.state_dict(), sft_predictor_pth)

sft_model.load_state_dict(torch.load(sft_predictor_pth, map_location="cpu"))
eval_sft()

class PPOPredictor(nn.Module):
    def __init__(self, bert_model, hidden_dim=768, num_actions=2):
        super().__init__()
        self.bert = bert_model
        self.fc = nn.Linear(hidden_dim, num_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(pooled_output)
        return self.softmax(logits)

ppo_model = PPOPredictor(bert_model, hidden_dim=768, num_actions=2)
sft_dict = torch.load(sft_predictor_pth, map_location="cpu")
ppo_model.load_state_dict(sft_dict, strict=False)

class PPOTrainer:
    def __init__(self, policy_model, lr=1e-5, epsilon=0.2, update_epochs=5):
        self.policy = policy_model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.epsilon = epsilon
        self.update_epochs = update_epochs
        self.memory = []

    def select_action(self, text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        with torch.no_grad():
            action_probs = self.policy(**inputs)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def store_transition(self, transition):
        # memory sliding window: only store 20 samples
        self.memory.append(transition)
        if len(self.memory) > 20:
            self.memory.pop(0) 


    def update_policy(self):
        if len(self.memory) < 10:
            return
        
        states, actions, rewards, old_log_probs = zip(*self.memory)
        inputs = tokenizer(list(states), return_tensors="pt", truncation=True, padding=True, max_length=1024)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)

        for _ in range(self.update_epochs):
            probs = self.policy(**inputs)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            advantages = rewards
            loss_policy = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            self.optimizer.zero_grad()
            loss_policy.backward()
            self.optimizer.step()

ppo_trainer = PPOTrainer(policy_model=ppo_model, lr=1e-5, epsilon=0.2, update_epochs=5)
import matplotlib.pyplot as plt
import collections
import time

N = 5
sliding_window_acc_values = collections.deque(maxlen=N)

acc_values = []
log_file = f"OpenAGI/stepwise/sft_ppo_distilbert_log/ma_adj_{N}_avg_acc_1.jsonl"
num_ppo_episodes = 1

for epoch in range(num_ppo_episodes):
    start_time = time.time()
    test_data = list(full_dataset)
    random.shuffle(test_data)
    with open(log_file, "w") as f:
        # accumulative_sample_num = 0
        # total_correct = 0
        for sample in test_data:
            text = sample["input"]
            choice = sample["output"]
            action, old_log_prob = ppo_trainer.select_action(text)
            reward = 1.0 if action == choice else -1.0
                
            # if reward == 1.0:
            #         total_correct += 1
            # accumulative_sample_num += 1 
            # accumulative_acc = round(total_correct / accumulative_sample_num, 3)
            # acc_values.append(accumulative_acc)
            sliding_window_acc_values.append(1 if reward > 0 else 0)
            sliding_window_avg_acc = round(sum(sliding_window_acc_values) / N, 3)
            
            log_data = {
                    # "task": state_text,
                    "ground_truth": "True" if choice == 1 else "False",
                    "predict continue": "True" if action == 1 else "False",
                    # "accumulative acc": accumulative_acc
                    f"adjcent_{N}_samples_acc": sliding_window_avg_acc
                    # "reward": reward
                }
            f.write(json.dumps(log_data) + "\n")
            ppo_trainer.store_transition((text, action, reward, old_log_prob))
        ppo_trainer.update_policy()
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch} Finished. Run for: {epoch_time:.2f} sec")

# plt.plot(acc_values)
# plt.xlabel("Test Samples")
# plt.ylabel("Accumulative Accuracy")
# plt.title("PPO Training Accuracy Over Time")
# plt.show()
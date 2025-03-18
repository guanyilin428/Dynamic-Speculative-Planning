import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
import random

def load_dataset(json_file):
    with open(json_file, "r") as f:
        dataset = json.load(f)
    return dataset

full_dataset = load_dataset("dataset_predict_k.json")

tokenizer = AutoTokenizer.from_pretrained("./weights/distilbert_base_uncased")
model = AutoModel.from_pretrained("./weights/distilbert_base_uncased")

def encode_text(state_text):
    inputs = tokenizer(state_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)  # shape: (768,)

class SharedPredictor(nn.Module):
    def __init__(self, hidden_dim=768, num_actions=6):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 输出 logits (SFT) 或者 logits+softmax (PPO)

class SFTPredictor(nn.Module):
    def __init__(self, hidden_dim=768, num_actions=6):
        super().__init__()
        self.mlp = SharedPredictor(hidden_dim, num_actions)  # 共享 MLP 结构

    def forward(self, x):
        return self.mlp(x)  # 直接输出 logits 供 CrossEntropyLoss 计算

sft_model = SFTPredictor(hidden_dim=768, num_actions=6)
sft_model.train()

def collate_fn(batch):
    embeddings, labels = [], []
    for sample in batch:
        text_input = sample["input"]
        k_label = int(sample["output"]) - 1

        emb = encode_text(text_input)
        embeddings.append(emb.unsqueeze(0))
        labels.append(k_label)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return embeddings, labels

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(sft_model.parameters(), lr=1e-4)

def train_sft(num_epochs=2):
    for epoch in range(num_epochs):
        sft_model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for embeddings, labels in train_loader:
            logits = sft_model(embeddings)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * embeddings.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += embeddings.size(0)

        train_acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2%}")

        sft_model.eval()
        with torch.no_grad():
            val_correct, val_samples = 0, 0
            for embeddings, labels in val_loader:
                logits = sft_model(embeddings)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += embeddings.size(0)
            val_acc = val_correct / val_samples
        print(f"Val Acc: {val_acc:.2%}")

predictor_weights_pth = "OpenAGI/predict_k/predictor_weights/sft_predictor.pth"
train_sft(num_epochs=30)
torch.save(sft_model.state_dict(), predictor_weights_pth)

class PPOPredictor(nn.Module):
    def __init__(self, hidden_dim=768, num_actions=6):
        super().__init__()
        self.mlp = SharedPredictor(hidden_dim, num_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        logits = self.mlp(x)
        return self.softmax(logits)

ppo_model = PPOPredictor(hidden_dim=768, num_actions=6)
sft_dict = torch.load(predictor_weights_pth, map_location="cpu")

new_sft_dict = {}
for k, v in sft_dict.items():
    new_k = k.replace("mlp.", "")
    new_sft_dict[new_k] = v

ppo_model.mlp.load_state_dict(new_sft_dict)

nn.init.xavier_uniform_(ppo_model.mlp.fc3.weight)
nn.init.zeros_(ppo_model.mlp.fc3.bias)


from torch.distributions import Categorical

class PPOTrainer:
    def __init__(self, policy_model, lr=1e-5, epsilon=0.2, update_epochs=3):
        self.policy = policy_model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.epsilon = epsilon
        self.update_epochs = update_epochs
        self.memory = []

    def select_action(self, state_tensor):
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()

    def store_transition(self, transition):
        # (state_vec, action_idx, reward, old_log_prob)
        self.memory.append(transition)

    def update_policy(self):
        if len(self.memory) < 10:
            return
        
        states, actions, rewards, old_log_probs = zip(*self.memory)
        states = torch.stack(states, dim=0)  # [batch_size, 768]
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)

        for _ in range(self.update_epochs):
            probs = self.policy(states)  # [batch_size, 6]
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            advantages = rewards

            loss_policy = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            self.optimizer.zero_grad()
            loss_policy.backward()
            self.optimizer.step()

        self.memory.clear()


ppo_trainer = PPOTrainer(policy_model=ppo_model, lr=1e-5, epsilon=0.2, update_epochs=3)

# breakpoint()
log_file = "OpenAGI/predict_k/sft_ppo_mlp_log.jsonl"
num_ppo_episodes = 1
for ep in range(num_ppo_episodes):
    test_data = list(val_dataset)
    total_correct = 0
    total_samples = len(test_data)

    with open(log_file, "w") as f:    
        accumulative_sample_num = 0
        for sample in test_data:
            text = sample["input"]
            ground_truth_k = sample["output"]
            # encode state
            emb = encode_text(text)  # shape [768]
            emb = emb.unsqueeze(0)   # [1,768]

            action_idx, old_log_prob = ppo_trainer.select_action(emb)
            pred_k = action_idx + 1
            if pred_k == ground_truth_k:
                    reward = 20.0
            elif pred_k == ground_truth_k + 1:
                reward = 5.0
            elif pred_k == ground_truth_k - 1:
                reward = -1
            else: 
                reward = -10 * abs(ground_truth_k - pred_k)
                
            if reward > 2:
                    total_correct += 1
            accumulative_sample_num += 1 
            
            log_data = {
                    # "task": state_text,
                    "ground_truth_k": ground_truth_k,
                    "predicted_k": pred_k,
                    "accumulative acc": round(total_correct / accumulative_sample_num, 3)
                    # "reward": reward
                }
            f.write(json.dumps(log_data) + "\n")
                
            ppo_trainer.store_transition((emb.squeeze(0), action_idx, reward, old_log_prob))

        ppo_trainer.update_policy()
        acc = total_correct / total_samples
        print(f"[PPO Episode {ep+1}] Acc: {acc:.2%}")

import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel
import random


def load_dataset(json_file):
    with open(json_file, "r") as f:
        dataset = json.load(f)
    return dataset

dataset = load_dataset("dataset_predict_k_cot.json")


tokenizer = AutoTokenizer.from_pretrained("./weights/distilbert_base_uncased")
model = AutoModel.from_pretrained("./weights/distilbert_base_uncased")

def encode_state_with_transformer(state_text):
    inputs = tokenizer(state_text, return_tensors="pt", truncation=True,
                       padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


class PPOPredictor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PPOPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

class PPOValueNet(nn.Module):
    def __init__(self, input_dim):
        super(PPOValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PPOTrainer:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=0.2, update_epochs=5):
        self.policy = PPOPredictor(state_dim, action_dim)
        self.value_net = PPOValueNet(state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.memory = deque(maxlen=5000)

        self.gamma = gamma
        self.epsilon = epsilon
        self.update_epochs = update_epochs
        self.action_dim = action_dim  
    
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return (action.item() + 1), log_prob.item()

    def store_transition(self, transition):
        # transition: (state, action_k, reward, log_prob, next_state)
        self.memory.append(transition)

    def update_policy(self):
        if len(self.memory) < 10:
            return

        states, actions, rewards, old_log_probs, next_states = zip(*self.memory)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        actions = [a - 1 for a in actions]
        actions = torch.tensor(actions, dtype=torch.long)

        for _ in range(self.update_epochs):
            action_probs = self.policy(states)  # shape: [batch_size, action_dim]
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            values = self.value_net(states).squeeze(-1)  # shape: [batch_size]

            advantages = rewards - values.detach()

            ratio = torch.exp(new_log_probs - old_log_probs)

            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

            loss_policy = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            loss_value = (values - rewards).pow(2).mean()

            loss = loss_policy + loss_value

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

        self.memory.clear()


state_dim = 768
action_dim = 6

trainer = PPOTrainer(state_dim, action_dim, update_epochs=3)
log_file = "ppo_training_log_3.jsonl"

num_episodes = 1
for episode in range(num_episodes):
    random.shuffle(dataset)
    total_samples = len(dataset)

    with open(log_file, "a") as f:
        total_correct = 0
        accumulative_sample_num = 0
        for sample in dataset:
            state_text = sample["input"]
            ground_truth_k = sample["output"]
            state_vec = encode_state_with_transformer(state_text)

            action_k, log_prob = trainer.select_action(state_vec)
            
            if action_k == ground_truth_k:
                reward = 20.0
            elif action_k == ground_truth_k + 1:
                reward = 5.0
            elif action_k == ground_truth_k - 1:
                reward = -1
            else: 
                reward = -10 * abs(ground_truth_k - action_k)
            
            if reward > 2:
                total_correct += 1
            accumulative_sample_num += 1 
            log_data = {
                # "task": state_text,
                "ground_truth_k": ground_truth_k,
                "predicted_k": action_k,
                "accumulative acc": round(total_correct / accumulative_sample_num, 3)
                # "reward": reward
            }
            f.write(json.dumps(log_data) + "\n")

            trainer.store_transition((state_vec, action_k, reward, log_prob, state_vec))

        trainer.update_policy()

        acc = total_correct / total_samples
        print(f"[Episode {episode+1}] Accuracy: {acc:.2%}")


def evaluate(trainer, dataset):
    total_correct = 0
    total_samples = len(dataset)
    for sample in dataset:
        state_text = sample["input"]
        ground_truth_k = sample["output"]
        state_vec = encode_state_with_transformer(state_text)
        
        action_k, _ = trainer.select_action(state_vec)
        if action_k == ground_truth_k:
            total_correct += 1
    accuracy = total_correct / total_samples
    print(f"Evaluation Accuracy: {accuracy:.2%}")

evaluate(trainer, dataset)

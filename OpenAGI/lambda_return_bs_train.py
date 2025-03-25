import json
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import argparse
import random
from torch.utils.data.sampler import Sampler

logging.basicConfig(
    filename='lambda_return_react_log_1_1.json',
    level=logging.INFO,
    filemode='w'
)

class DistilBERTValueFunction(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.model = bert_model
        self.fc = nn.Linear(self.model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        k = self.fc(cls_embedding)
        # k = self.relu(k)
        return k

class LambdaReturnDataset(Dataset):
    def __init__(self, episodes, tokenizer, model, max_length=512):
        self.episodes = episodes
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model = model
        self.trajectory_lengths = [len(episode['trajectory']) for episode in episodes]
    
    def __len__(self):
        return len(self.episodes)    
    
    def __getitem__(self, idx):
        trajectory = self.episodes[idx]['trajectory']
        
        states = []
        rewards = []   
        gt_ks = []           
        for step in trajectory:
            state = step['state']
            reward = step['reward']
            gt_k = step['k']
            
            states.append(state)
            rewards.append(torch.tensor(reward))
            gt_ks.append(torch.tensor(gt_k))

        inputs = tokenizer(
            states, 
            return_tensors="pt", 
            truncation=True, 
            padding='max_length', 
            max_length=512
        )
        input_ids = inputs['input_ids']
        attention_masks = inputs['attention_mask']
        rewards = torch.stack(rewards, dim=0)
        gt_ks = torch.stack(gt_ks, dim=0)

        return input_ids, attention_masks, rewards, gt_ks
    


class DynamicBatchSampler(Sampler):
    def __init__(self, trajectory_lengths, max_steps_per_batch, shuffle=True):
        self.trajectory_lengths = trajectory_lengths
        self.max_steps = max_steps_per_batch
        self.shuffle = shuffle
        
    def __iter__(self):
        indices = list(range(len(self.trajectory_lengths)))
        if self.shuffle:
            random.shuffle(indices)
            
        batch = []
        current_steps = 0
        for idx in indices:
            traj_len = self.trajectory_lengths[idx]
            
            if current_steps + traj_len > self.max_steps and batch:
                yield batch
                batch = [idx]
                current_steps = traj_len
            else:
                batch.append(idx)
                current_steps += traj_len
        
        if batch:
            yield batch
            
    def __len__(self):
        count = 0
        current = 0
        for length in self.trajectory_lengths:
            if current + length > self.max_steps:
                count += 1
                current = length
            else:
                current += length
        if current > 0:
            count += 1
        return count

def collate_fn(batch):
    input_ids_batch = []
    attention_mask_batch = []
    rewards_batch = []
    gt_k_batch = []
    for trajectory_data in batch:
        input_ids_batch.append(trajectory_data[0])
        attention_mask_batch.append(trajectory_data[1])
        rewards_batch.append(trajectory_data[2])
        gt_k_batch.append(trajectory_data[3])
    return input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch


def compute_lambda_return(input_ids, attention_masks, rewards, gamma, lambda_, model):
    model.eval()
    T = len(input_ids)
    G_lambda = torch.zeros(T)
    
    G_t = 0
    for t in reversed(range(T)):
        input_id = input_ids[t].unsqueeze(0)
        attention_mask = attention_masks[t].unsqueeze(0)
        v_pred = model(input_id, attention_mask).item()   
        mask = rewards[t]
        G_t = rewards[t] + gamma * (1 - lambda_) * v_pred * mask + gamma * lambda_ * G_t * mask
        G_lambda[t] = G_t
    model.train()
    return G_lambda

def flat_batch_data(input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch, gamma, lambda_, model):
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
        G_lambda = compute_lambda_return(input_ids, attention_mask, rewards, gamma, lambda_, model)
        # flatten inputs and targets
        all_G_lambda.extend(G_lambda) 
        all_input_ids.extend(input_ids)
        all_attention_mask.extend(attention_mask)
        all_gt_k.extend(gt_k_batch[i])
        
    device = model.model.device
    # [total_step_num, hid_dim]: total_step_num in this batch
    flat_input_ids = torch.stack(all_input_ids, dim=0).to(device)
    flat_attention_mask = torch.stack(all_attention_mask, dim=0).to(device)
    # [total_step_num]
    flat_G_lambda = torch.tensor(all_G_lambda).to(device)
    flat_gt_k = torch.stack(all_gt_k, dim=0).to(device)
    return flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k


def train_model(train_dataloader, val_dataloader, gamma, lambda_, model, tokenizer, writer, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    num_epochs = 35
    for epoch in range(num_epochs):
        model.train()
        epoch_acc = 0.0
        epoch_loss = 0.0
        total_batches = len(train_dataloader)
        
        # logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, (input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch) in enumerate(train_dataloader):
                            
            flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k = flat_batch_data(input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch, gamma, lambda_, model)

            k_pred = model(flat_input_ids, flat_attention_mask).squeeze()
            loss = criterion(k_pred, flat_G_lambda)
            diff = torch.round(k_pred) - flat_gt_k
            acc = ((diff == 0) | (diff == 1)).float().mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

            writer.add_scalar('Loss/train', loss.item(), epoch * total_batches + batch_idx)
            writer.add_scalar('Accuracy/train', acc, epoch * total_batches + batch_idx)

            # logging.info(f"Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

        epoch_acc /= total_batches
        epoch_loss /= total_batches
        # logging.info(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}, Average Accuracy: {epoch_acc:.4f}")
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        writer.add_scalar('Accuracy/epoch', epoch_acc, epoch)

        eval_model(val_dataloader, criterion, gamma, lambda_, model, tokenizer, epoch, writer)
        

def eval_model(val_dataloader, criterion, gamma, lambda_, model, tokenizer, epoch, writer):
    model.eval()
    
    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0.0
        total_batches = len(val_dataloader)
        for input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch in val_dataloader:
            
            flat_input_ids, flat_attention_mask, flat_G_lambda, flat_gt_k = flat_batch_data(input_ids_batch, attention_mask_batch, rewards_batch, gt_k_batch, gamma, lambda_, model)
            
            k_preds = model(flat_input_ids, flat_attention_mask).squeeze()
            loss = criterion(k_preds, flat_G_lambda)
            diff = torch.round(k_preds) - flat_gt_k
            acc = ((diff == 0) | (diff == 1)).float().mean().item()

            val_loss += loss.item()
            val_acc += acc

    val_loss /= total_batches
    val_acc /= total_batches

    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)


def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with custom batch_size and learning_rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--lambda_", type=float, default=1, help="Lambda for lambda return.")
    
    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    lambda_ = args.lambda_
    gamma = 1
    
    ds_type = "cot"
    
    if ds_type == "cot":
        dataset_path = "dataset_value_func_cot_new.json"
    elif ds_type == "tp":
        dataset_path = "dataset_value_func_react.json"
    
    model_path = "distilbert-base-uncased"
    bert_model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DistilBERTValueFunction(bert_model)
    
    episodes = load_data_from_json(dataset_path)
    train_episodes, val_episodes = train_test_split(episodes, test_size=0.2, random_state=42)
    train_dataset = LambdaReturnDataset(train_episodes, tokenizer, model)
    val_dataset = LambdaReturnDataset(val_episodes, tokenizer, model)
    
    train_batch_sampler = DynamicBatchSampler(
        trajectory_lengths=train_dataset.trajectory_lengths,
        max_steps_per_batch=batch_size,
        shuffle=True
    )
    val_batch_sampler = DynamicBatchSampler(
        trajectory_lengths=val_dataset.trajectory_lengths,
        max_steps_per_batch=batch_size,
        shuffle=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=collate_fn,
        pin_memory=True
    ) 
    
    writer = SummaryWriter(log_dir=f'test/dyn_bs_lambda_{lambda_}_{ds_type}/lr_{lr}/bs_{batch_size}')
    # writer = SummaryWriter(log_dir=f'test/dyn_bs_lambda_{lambda_}_{ds_type}/lr_{lr}/bs_{batch_size}')
    train_model(train_dataloader, val_dataloader, gamma=gamma, lambda_=lambda_, model=model, tokenizer=tokenizer, writer=writer, lr=lr)
    model_save_path = "cot_value_function_model.pth"
    torch.save(model.state_dict(), model_save_path)
    # logging.info(f"Model weights saved to {model_save_path}")

import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
import random
from torch.distributions import Categorical
import collections
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig
import argparse

def load_dataset(json_file):
    with open(json_file, "r") as f:
        dataset = json.load(f)
    return dataset


class SFTPredictor(nn.Module):
    def __init__(self, bert_model, num_actions=1):
        super().__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size, num_actions)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        k = self.fc(pooled_output)
        return k
    

def collate_fn(batch):
    input_texts, labels = [], []
    for sample in batch:
        input_texts.append(sample["input"])
        labels.append(int(sample["output"]) - 1)
    
    inputs = tokenizer(
        input_texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    labels = torch.tensor(labels, dtype=torch.float32)
    return inputs, labels

def train_sft(train_loader, val_loader, model, lr, writer):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()        
        epoch_acc = 0
        epoch_loss = 0
        total_batches = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            k_preds = model(**inputs).squeeze()
            loss = criterion(k_preds, labels)
            diff = torch.round(k_preds) - labels
            acc = ((diff == 0) | (diff == 1)).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            
            writer.add_scalar('Loss/train', loss.item(), epoch * total_batches + batch_idx)
            writer.add_scalar('Accuracy/train', acc, epoch * total_batches + batch_idx)
   
        epoch_acc /= total_batches
        epoch_loss /= total_batches
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        writer.add_scalar('Accuracy/epoch', epoch_acc, epoch)

        eval_sft(val_loader, criterion, epoch, model, writer)
        
def eval_sft(val_loader, criterion, epoch, model, writer):
    model.eval()
    with torch.no_grad():
        val_acc, val_loss = 0, 0
        total_batches = len(val_loader)
        
        for inputs, labels in val_loader:
            k_preds = model(**inputs).squeeze()
            loss = criterion(k_preds, labels)
            diff = torch.round(k_preds) - labels
            acc = ((diff == 0) | (diff == 1)).float().mean().item()
            
            val_loss += loss.item()
            val_acc += acc
            
        val_acc /= total_batches
        val_loss /= total_batches
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

if __name__ == "__main__":    
    # dataset_predict_k_cot.json
    # dataset_travelplanner_first_task.json
    parser = argparse.ArgumentParser(description="Train the model with custom batch_size and learning_rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer.")

    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    
    model_path = "../weights/distilbert_base_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    bert_model = AutoModel.from_pretrained(model_path, local_files_only=True)
    
    full_dataset = load_dataset("dataset_value_to_sft_cot.json")
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    writer = SummaryWriter(log_dir=f"3_18_log/sft_reg_cot_val/lr_{lr}_bs_{batch_size}")

    # sft_predictor_pth = "./OpenAGI/predict_k/predictor_weights/sft_distilbert_predictor.pth"
    sft_predictor_pth = "./OpenAGI/predict_k/predictor_weights/sft_travel_planner.pth"
    sft_model = SFTPredictor(bert_model, num_actions=1)
    
    train_sft(train_loader, val_loader, sft_model, lr, writer)
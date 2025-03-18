import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AdamW
from sklearn.model_selection import train_test_split

class TrajectoryDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512, split="train"):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)

        self.data = train_data if split == "train" else test_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        label = torch.tensor(item["output"], dtype=torch.long)
        return {**inputs, "labels": label}


def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy


def train(json_path, model_name, epochs, batch_size=8, lr=5e-5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 假设最多10个类别
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    
    breakpoint()
    train_dataset = TrajectoryDataset(json_path, tokenizer, max_length=512, split="train")
    test_dataset = TrajectoryDataset(json_path, tokenizer, max_length=512, split="test")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            batch = {key: val.to(device) for key, val in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

    model.save_pretrained("./distilbert_stepwise_model")
    tokenizer.save_pretrained("./distilbert_stepwise_model")
    print("Model Saved.")

    test_accuracy = evaluate_accuracy(model, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    dataset_path = "dataset_stepwise.json"
    train(json_path=dataset_path, 
          model_name="../weights/distilbert_base_uncased", 
          epochs=30, 
          batch_size=8)
    # test_accuracy = evaluate_accuracy(model, test_dataloader, device)
    # print(f"Test Accuracy: {test_accuracy:.4f}")

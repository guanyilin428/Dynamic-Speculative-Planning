import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch

class TrajectoryDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512, split="train"):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # Split the data into training and validation sets
        train_data, val_data = train_test_split(self.data, test_size=0.2, random_state=42)
        
        # Use the appropriate split based on the argument
        self.data = train_data if split == "train" else val_data

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


def train(json_path, model_name, epochs, batch_size=8, lr=5e-5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10)  # 假设最多10个输出类别

    train_dataset = TrajectoryDataset(json_path, tokenizer, max_length=512, split="train")
    val_dataset = TrajectoryDataset(json_path, tokenizer, max_length=512, split="val")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {key: val.to(device) for key, val in batch.items()}

                outputs = model(**batch)
                val_loss += outputs.loss.item()

        print(f"Validation Loss after epoch {epoch + 1}: {val_loss / len(val_dataloader)}")

    model.save_pretrained("./distilbert_pred_k_model")
    tokenizer.save_pretrained("./distilbert_pred_k_model")
    print("Model Saved.")


if __name__ == "__main__":
    dataset_path = "dataset_with_trajectory.json"
    train(json_path=dataset_path, \
          model_name="weights/distilbert_base_uncased", \
          epochs=50,
          batch_size=8)
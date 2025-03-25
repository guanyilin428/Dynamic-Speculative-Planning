import torch.nn as nn

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
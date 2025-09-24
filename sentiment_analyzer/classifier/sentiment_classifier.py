import json
import torch
import torch.nn as nn
from transformers import DistilBertModel

# Load config file (make sure config.json has "BERT_MODEL": "distilbert-base-uncased")
with open("config.json") as f:
    config = json.load(f)

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super(SentimentClassifier, self).__init__()
        # Use DistilBERT instead of full BERT
        self.bert = DistilBertModel.from_pretrained(config["BERT_MODEL"])
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # DistilBERT returns last_hidden_state, not pooled_output like BERT
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        pooled_output = hidden_state[:, 0]       # take [CLS] token representation
        dropped = self.drop(pooled_output)
        return self.out(dropped)

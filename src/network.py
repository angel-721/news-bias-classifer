import torch
import torch.nn as nn
from transformers import DistilBertModel


class NetworkClassifer(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.at = nn.Linear(768, 1)
        self.fc = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_ids, mask):
        # DistilBERT outputs float embeddings: (batch, seq_len, 768)
        outputs = self.bert(input_ids=input_ids, attention_mask=mask)
        x = outputs.last_hidden_state

        scores = self.at(x)
        # mask is 1 for real tokens, 0 for padding — invert for masked_fill
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * x).sum(dim=1)

        return self.fc(pooled)

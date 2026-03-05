from config import LABELS
import polars as pl
import torch
import warnings
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

# Suppress tokenizer sequence length warning (we handle chunking manually)
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


class BiasDataset(Dataset):
    def __init__(self, path, window_size=512, stride=256):
        self.window_size = window_size
        self.stride = stride

        df = pl.read_parquet(path)
        texts = df["content"].str.replace_all("\n", " ").to_list()
        labels = [LABELS[b] for b in df["text_label"].to_list()]

        self.chunks = []  
        for text, label in zip(texts, labels):
            encoded = tokenizer(
                text,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)

            # Create sliding windows
            for start_idx in range(0, len(input_ids), self.stride):
                end_idx = min(start_idx + self.window_size, len(input_ids))
                chunk_ids = input_ids[start_idx:end_idx]

                attention_mask = torch.ones_like(chunk_ids)
                if len(chunk_ids) < self.window_size:
                    pad_length = self.window_size - len(chunk_ids)
                    chunk_ids = torch.cat([
                        chunk_ids,
                        torch.zeros(pad_length, dtype=chunk_ids.dtype)
                    ])
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.zeros(pad_length, dtype=attention_mask.dtype)
                    ])

                self.chunks.append((chunk_ids, attention_mask, label))

                if end_idx == len(input_ids):
                    break

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        input_ids, attention_mask, label = self.chunks[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label),
        }

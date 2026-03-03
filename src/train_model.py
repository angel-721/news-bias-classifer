import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

bias_map = {
    "left": 0,
    "leaning-left": 1,
    "center": 2,
    "leaning-right": 3,
    "right": 4,
}


class CustomChunckedDataSet(Dataset):
    def __init__(
        self,
        embeddings_file,
    ):
        self.df: pl.DataFrame = pl.read_parquet(embeddings_file)
        chunks_list = [
            torch.tensor(chunks) for chunks in self.df["chunked_embeddings"].to_list()
        ]

        self.padded = pad_sequence(chunks_list, batch_first=True)

        # Create the attenion mask for articles where embedding chunks might be padded
        chunk_lengths = [
            len(chunks) for chunks in self.df["chunked_embeddings"].to_list()
        ]
        self.mask = torch.zeros(
            len(chunks_list), self.padded.shape[1], dtype=torch.bool
        )
        for i, length in enumerate(chunk_lengths):
            self.mask[i, :length] = True

        self.labels = torch.tensor([bias_map[b] for b in self.df["bias"].to_list()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.padded[idx], self.mask[idx], self.labels[idx]


# data_set = CustomChunckedDataSet("./embeddings/labeled_embeddings.parquet")
# I should probably split this data soon

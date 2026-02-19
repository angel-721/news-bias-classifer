import pickle

import kagglehub
import polars as pl
from sentence_transformers import SentenceTransformer

path = kagglehub.dataset_download("devdope/900k-spotify")
filePath = f"{path}/spotify_dataset.csv"
df: pl.DataFrame = pl.read_csv(filePath)

songs = df["text"].to_numpy()

model: SentenceTransformer = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
)
model.max_seq_length = 256  # already default, but forces truncation

embeddings = model.encode(songs, batch_size=256, show_progress_bar=True)

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

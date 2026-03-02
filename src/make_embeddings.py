import os
import sys

import kagglehub
import polars as pl
from sentence_transformers import SentenceTransformer

# MAKE SURE TO SET A HF_TOKEN!!!!!! :)


def makeChunks(text: list[list[str]]) -> list[list[str]]:
    # I'll assume that a chunk is around 256 tokens or a 190 words
    chunks = []
    for i in range(0, len(text), 190):
        chunks.append(text[i : i + 190])

    return chunks


def makeEmbeddingsAndSave(saveData: bool = False):
    # Just use linux lol
    path = "~/.cache/kagglehub/datasets/gandpablo/news-articles-for-political-bias-classification/versions/1"
    if not os.path.isdir(path):
        path = kagglehub.dataset_download(
            "gandpablo/news-articles-for-political-bias-classification"
        )

    file_path = f"{path}/bias_clean.csv"

    df: pl.DataFrame = pl.read_csv(file_path)

    # clean text for chunking and emedding
    df = df.with_columns(pl.col("page_text").str.replace_all("\n", " ").str.split(" "))

    df = df.with_columns(
        pl.col("page_text").map_elements(makeChunks).alias("article_chunks")
    )

    all_chunks = df["article_chunks"].to_list()
    chunk_counts = [len(chunks) for chunks in all_chunks]
    flat_sentences = [" ".join(chunk) for chunks in all_chunks for chunk in chunks]

    model: SentenceTransformer = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cuda"
    )
    all_embeddings = model.encode(
        flat_sentences, batch_size=256, show_progress_bar=True
    ).tolist()

    # Split embeddings back per article using chunk counts
    split_embeddings = []
    idx = 0
    for count in chunk_counts:
        split_embeddings.append(all_embeddings[idx : idx + count])
        idx += count

    labeled_embeddings = df.with_columns(
        pl.Series("chunked_embeddings", split_embeddings)
    ).select(["url", "chunked_embeddings", "bias"])

    print(labeled_embeddings.head())

    if saveData:
        labeled_embeddings.write_parquet("./embeddings/labeled_embeddings.parquet")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""need arg for saving data. E.g
              uv run src/make_embeddings.py --save true
              uv run src/make_embeddings.py --save false
              """)
    else:
        makeEmbeddingsAndSave(sys.argv[1].lower() == "true")

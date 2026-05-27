# scripts/02_embed.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Constants
INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = f"{OUTPUT_DIR}/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64


def main():
    # Load dataset
    df = pd.read_parquet(INPUT_FILE)

    # Prepare texts for embedding
    texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()
    print(texts[0])

    # Load the embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )

    # Results
    print(f"Total number of processed texts: {len(texts)}")
    print(f"The dimensions of the embeddings array: {embeddings.shape}")
    print(f"L2 norm of the first embedding: {np.linalg.norm(embeddings[0]):.4f}")

    # Save embeddings
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)


if __name__ == "__main__":
    main()

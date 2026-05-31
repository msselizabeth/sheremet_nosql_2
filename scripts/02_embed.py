# scripts/02_embed.py
import os
import numpy as np
import pandas as pd
import torch
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
    # print(texts[0])

    # Load the embedding model
    # model = SentenceTransformer(MODEL_NAME)
    
    # Check for Apple Silicon Metal Performance Shaders (MPS) availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        # Fallback for older Intel Macs
        device = torch.device("cpu")

    # Apple Silicon GPUs are highly optimized for float16. 
    # Avoid bfloat16 on MPS as it lacks native hardware support.
    model_args = {"torch_dtype": torch.float16}

    # Initialize the model and map it to the selected device
    model = SentenceTransformer(MODEL_NAME, model_kwargs=model_args).to(device)

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

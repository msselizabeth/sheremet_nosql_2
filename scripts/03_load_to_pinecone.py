import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200


def main():
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Create index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # Get Index 
    index = pc.Index(INDEX_NAME)

    # Load data
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)

    # Prepare and upload batches
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        # Get the batch slices
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_embs = embeddings[i : i + BATCH_SIZE]

        vectors_to_upsert = []

        # Process each record in the current batch
        for j in range(len(batch_df)):
            absolute_idx = i + j  # Absolute index across the dataset
            row = batch_df.iloc[j]

            vector_id = f"paper_{absolute_idx}"

            metadata = {
                "arxiv_id": row["id"],
                "title": row["title"],
                "abstract": row["abstract"][:500],
                "authors": row["authors"][:200],
                "year": int(row["year"]),
                "category": row["category"],
            }

            vectors_to_upsert.append(
                {
                    "id": vector_id,
                    "values": batch_embs[j].tolist(),
                    "metadata": metadata,
                }
            )

        index.upsert(vectors=vectors_to_upsert)

    # Final statistics
    stats = index.describe_index_stats()
    print(f"Total vectors uploaded: {stats.total_vector_count}")


if __name__ == "__main__":
    main()

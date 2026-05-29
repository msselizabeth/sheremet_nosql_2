# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# Constants
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
LOCAL_PARQUET = "data/arxiv_subset.parquet"
LOCAL_EMBEDDINGS = "embeddings/embeddings.npy"


def get_query_embedding(query: str, model: SentenceTransformer) -> list[float]:
    """
    Encodes the search query into a normalized embedding vector.
    """

    query_embedding = model.encode(query, normalize_embeddings=True)
    return query_embedding.tolist()


def print_pinecone_results(response: dict, title: str) -> None:
    """Helper to format and print Pinecone search results."""

    print(f"\n{'-'*35}\n{title}\n{'-'*35}")

    if len(response["matches"]) == 0:
        print(f"No Results Found.")

    for i, item in enumerate(response["matches"], 1):
        metadata = item.get("metadata", {})
        print(f"Top_{i}: {metadata.get('title', 'No Title')}")
        print(
            f"Year: {metadata.get('year', 'N/A')} | Category: {metadata.get('category', 'N/A')}"
        )
        print(f"Abstract: {metadata.get('abstract', '')[:120]}...\n")


def print_local_results(indices: np.ndarray, df: pd.DataFrame, title: str) -> None:
    """Helper to print top results from local numpy calculations."""

    if len(indices) == 0:
        print(f"No Results Found.")

    print(f"\n{'-'*35}\nLocal Results: {title}\n{'-'*35}")
    for i, item in enumerate(indices, 1):
        row = df.iloc[item]
        print(f"Top_{i}: {row['title']}")
        print(f"Year: {row['year']} | Category: {row['category']}")
        print(f"Abstarct: {row['abstract'][:120]}...\n")


def main():

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)

    model = SentenceTransformer(MODEL_NAME)
    df = pd.read_parquet(LOCAL_PARQUET)

    # ---------------------------------------  
    # Pure Semantic Search
    # ---------------------------------------

    query_1 = "teaching machines to recognize objects in pictures"
    query_1_embed = get_query_embedding(query_1, model)
    result = index.query(vector=query_1_embed, top_k=TOP_K, include_metadata=True)

    # Print Results
    print_pinecone_results(result, "Pure Semantic Search Results:")

    # ---------------------------------------
    # Search with Metadata Filtering
    # ---------------------------------------
    query_2 = "reinforcement learning"
    query_2_embed = get_query_embedding(query_2, model)

    # Filter A: Last 5 years (>= 2019) AND category "cs.LG"
    filter_a = {"$and": [{"category": {"$eq": "cs.LG"}}, {"year": {"$gte": 2019}}]}
    # Query for filter A
    results_a = index.query(
        vector=query_2_embed, top_k=5, include_metadata=True, filter=filter_a
    )

    # Filter B: Older papers (< 2015), any category
    filter_b = {"year": {"$lt": 2015}}
    # Query for filter B
    results_b = index.query(
        vector=query_2_embed, top_k=5, include_metadata=True, filter=filter_b
    )

    # Print Results
    print_pinecone_results(results_a, "Filter A: >= 2019 AND cs.LG")
    print_pinecone_results(results_b, "Filter B: < 2015")

    # ---------------------------------------
    # Local Metric Comparison
    # ---------------------------------------
    embeddings = np.load(LOCAL_EMBEDDINGS)
    query_emb_np = get_query_embedding(query_1, model)

    # Dot Product
    dot_scores = np.dot(embeddings, query_emb_np)
    # Take the last 5 elements (largest) and reverse them
    top_5_dot = np.argsort(dot_scores)[-5:][::-1]

    # Cosine Similarity = dot_scores / (norm(embeddings) * norm(query))
    cosine_scores = (
        dot_scores / np.linalg.norm(embeddings) * np.linalg.norm(query_emb_np)
    )
    top_5_cos = np.argsort(cosine_scores)[-5:][::-1]

    # L2 Distance (Smaller distance = more similar)
    l2_distances = np.linalg.norm(embeddings - query_emb_np, axis=1)
    top_5_l2 = np.argsort(l2_distances)[:5]

    # Print Resluts
    print_local_results(top_5_dot, df, "Top 5 Dot Product Results")
    print_local_results(top_5_cos, df, "Top 5 Cosine Results")
    print_local_results(top_5_l2, df, "Top 5 L2 Results")


if __name__ == "__main__":
    main()

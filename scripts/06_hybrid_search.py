# scripts/06_hybrid_search.py

import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

# Constants
INPUT_PARQUET = "data/arxiv_subset.parquet"
INDEX_NAME = "arxiv-papers"  
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10  


def prepare_bm25(df: pd.DataFrame) -> BM25Okapi:
    """
    Tokenizes titles and abstracts to build a local BM25 index.
    """

    corpus = (df["title"] + " " + df["abstract"]).astype(str).tolist()
    tokenized_corpus = [text.lower().split() for text in corpus]
    return BM25Okapi(tokenized_corpus)


def search_bm25(bm25: BM25Okapi, query: str, top_k: int = TOP_K) -> list[int]:
    """
    Executes lexical search and returns top_k matching document indices.
    """
    # TODO: Step 2 - Tokenize query, get scores, and return top_k indices sorted descending
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranking = np.argsort(bm25_scores)[::-1][:top_k]
    return bm25_ranking.tolist()


def search_dense(
    index, query: str, model: SentenceTransformer, top_k: int = TOP_K
) -> list[int]:
    """
    Executes vector search in Pinecone and returns top_k matching document indices (converted to integers).
    """

    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    if len(result["matches"]) == 0:
        print("No rESults Found")
        return []

    dense_ranking = []
    for match in result["matches"]:
        idx = int(match["id"].split("_")[1])
        dense_ranking.append(idx)

    return dense_ranking


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """
    Combines multiple ranked lists of document indices using RRF.
    Returns a sorted list of (doc_index, rrf_score) tuples.
    """
    scores: dict[int, float] = {}
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            # rank starts at 0
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            
    # Sort descending by RRF score
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def main():
    # Initialize clients and load data
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)

    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)

    # Setup BM25
    bm25_index = prepare_bm25(df)

    # Required test queries
    test_queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text",
    ]

    for query in test_queries:
        print(f"\nEvaluating query: '{query}'")

        # Execute searches
        bm25 = search_bm25(bm25_index, query, TOP_K)
        dense = search_dense(index, query, model, TOP_K)
        hybrid = reciprocal_rank_fusion([bm25, dense])

        # Generate and print a comparative Markdown table
        print("| Rank | BM25 | Pinecone | RRF |")
        print("|---|---|---|---|")

        # Print Results
        for i in range(5):
            t_bm25 = df.iloc[bm25[i]]['title']
            t_dense = df.iloc[dense[i]]['title']
            
            h_idx, h_score = hybrid[i]
            t_hybrid = f"{df.iloc[h_idx]['title']} *(Score: {h_score:.4f})*"
            
            print(f"| {i+1} | {t_bm25} | {t_dense} | {t_hybrid} |")


if __name__ == "__main__":
    main()

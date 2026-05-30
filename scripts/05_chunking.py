import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Constants
INPUT_PARQUET = "data/arxiv_subset.parquet"
MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
BATCH_SIZE = 200

INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"


def get_fixed_chunks(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    """
    Splits text into chunks of fixed word count with a specified overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text.strip())


def get_semantic_chunks(
    text: str,
    model: SentenceTransformer,
    threshold: float = 0.7,
    min_chunk_size: int = 50,
) -> list[str]:
    """
    Splits text into semantically cohesive chunks.
    A new chunk begins when the cosine similarity between adjacent sentences drops below the threshold.
    """
    # Split into sentences
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]

    if len(sentences) < 2:
        return sentences

    # Get normalized embeddings
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # Calculate similarities
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    chunks = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        # If similarity drops below threshold, we split
        if sim < threshold and len(" ".join(current_chunk).split()) >= min_chunk_size:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks


def init_pinecone_index(pc: Pinecone, index_name: str) -> None:
    """
    Creates a Pinecone index if it doesn't exist.
    """
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def process_and_upload(
    index, papers_df: pd.DataFrame, model: SentenceTransformer, chunk_strategy: str
):
    """
    Generates chunks for the dataset, embeds them, and uploads to Pinecone in batches.
    """

    vectors_to_upsert = []

    for _, row in tqdm(
        papers_df.iterrows(), total=len(papers_df), desc=f"Chunking ({chunk_strategy})"
    ):
        text = str(row["abstract"])
        paper_id = str(row["id"])

        # Chunk Strategy
        if chunk_strategy == "fixed":
            chunks = get_fixed_chunks(text)
        elif chunk_strategy == "semantic":
            chunks = get_semantic_chunks(text, model)
        else:
            raise ValueError(f"Unknown strategy: {chunk_strategy}")

        # Create embeds fo all chunks
        embeddings = model.encode(chunks, normalize_embeddings=True)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):

            metadata = {
                "arxiv_id": paper_id,
                "title": row["title"],
                "chunk_text": chunk,
                "chunk_index": i,
                "year": int(row["year"]),
                "category": row["category"],
            }

            vector_id = f"chunk_{paper_id}_{i}"

            vectors_to_upsert.append(
                {
                    "id": vector_id,
                    "values": emb.tolist(),
                    "metadata": metadata,
                }
            )

    for i in tqdm(range(0, len(vectors_to_upsert), BATCH_SIZE), desc="Upserting"):
        batch = vectors_to_upsert[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)


def search_chunks(index, query: str, model: SentenceTransformer, top_k: int = 5):
    """
    Queries the Pinecone index and prints results.
    """

    # Encode query and convert to list for Pinecone
    query_embed = model.encode(query, normalize_embeddings=True).tolist()

    # Execute Query
    result = index.query(vector=query_embed, top_k=top_k, include_metadata=True)

    if len(result["matches"]) == 0:
        print("No Results Found.")
        return

    for i, item in enumerate(result["matches"], 1):
        metadata = item.get("metadata", {})
        title = metadata.get("title", "No Title")
        chunk_idx = metadata.get("chunk_index", "N/A")
        chunk_text = metadata.get("chunk_text", "")

        print(f"Top {i} | Paper: {title} | Chunk #{chunk_idx}")
        print(f"Text: {chunk_text[:150]}...\n")


def main():

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    model = SentenceTransformer(MODEL_NAME)
    df = pd.read_parquet(INPUT_PARQUET)

    # ---------------------------------------------
    # Select 30 papers with the longest abstracts
    # ---------------------------------------------
    df["abstract_len"] = df["abstract"].str.len()
    top_30_papers = df.nlargest(30, "abstract_len").copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # STEP 2 & 3 & 4 & 5: Index Setup and Chunk Ingestion
    # ---------------------------------------------------------
    # Initialize indecies
    init_pinecone_index(pc, INDEX_FIXED)
    init_pinecone_index(pc, INDEX_SEMANTIC)

    fixed_index = pc.Index(INDEX_FIXED)
    semantic_index = pc.Index(INDEX_SEMANTIC)

    # Process and upload chunks for both strategies
    process_and_upload(fixed_index, top_30_papers, model, chunk_strategy="fixed")
    process_and_upload(semantic_index, top_30_papers, model, chunk_strategy="semantic")

    # ---------------------------------------------------------
    # Execute and Compare Test Queries
    # ---------------------------------------------------------
    test_queries = [
        "deep neural networks architecture",
        "statistical mechanics and algorithms",
    ]

    for query in test_queries:
        search_chunks(fixed_index, query, model)
        search_chunks(semantic_index, query, model)


if __name__ == "__main__":
    main()

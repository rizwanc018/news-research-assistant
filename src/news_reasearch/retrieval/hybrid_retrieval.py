import bm25s
from lancedb import Table
from sentence_transformers import SentenceTransformer

from news_reasearch.retrieval.bm25_retrieval import bm25_search
from news_reasearch.retrieval.semantic_retrieval import semantic_search


def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """Combine semantic and BM25 rankings using RRF."""

    fused_results = {}

    # Process semantic search results
    for rank, (_, row) in enumerate(
        semantic_results.iterrows(),
        start=1,
    ):
        chunk_id = row["chunk_id"]

        if chunk_id not in fused_results:
            fused_results[chunk_id] = {
                "chunk_id": chunk_id,
                "article_id": row["article_id"],
                "category": row["category"],
                "chunk": row["chunk"],
                "rrf_score": 0.0,
                "semantic_rank": None,
                "bm25_rank": None,
            }

        fused_results[chunk_id]["rrf_score"] += 1 / (k + rank)
        fused_results[chunk_id]["semantic_rank"] = rank

    # Process BM25 search results
    for rank, document in enumerate(
        bm25_results,
        start=1,
    ):
        chunk_id = document["chunk_id"]

        if chunk_id not in fused_results:
            fused_results[chunk_id] = {
                "chunk_id": chunk_id,
                "article_id": document["article_id"],
                "category": document["category"],
                "chunk": document["chunk"],
                "rrf_score": 0.0,
                "semantic_rank": None,
                "bm25_rank": None,
            }

        fused_results[chunk_id]["rrf_score"] += 1 / (k + rank)
        fused_results[chunk_id]["bm25_rank"] = rank

    # Sort by RRF score, highest first
    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:top_k]


def hybrid_search(
    query: str,
    model: SentenceTransformer,
    table: Table,
    bm25_retriever: bm25s.BM25,
    top_k: int = 10,
    candidate_k: int = 10,
    category: str | None = None,
) -> list[dict]:
    semantic_results = semantic_search(
        query=query,
        model=model,
        table=table,
        top_k=candidate_k,
        category=category,
    )

    bm25_results = bm25_search(
        query=query,
        retriever=bm25_retriever,
        top_k=candidate_k,
        category=category,
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
        k=60,
        top_k=top_k,
    )

    return fused_results

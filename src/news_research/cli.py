"""Command-line entry point for querying the news research index."""

import argparse

from news_research.embeddings import load_embedding_model
from news_research.retrieval.bm25_retrieval import load_bm25_index
from news_research.retrieval.hybrid_retrieval import hybrid_search
from news_research.vector_store import connect_to_lancedb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the BBC news corpus with hybrid retrieval.",
    )
    parser.add_argument("query", help="The search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Results to return.")
    parser.add_argument("--category", default=None, help="Filter by category.")
    args = parser.parse_args()

    db = connect_to_lancedb()
    table = db.open_table("news_chunks")
    model = load_embedding_model()
    bm25_retriever = load_bm25_index()

    results = hybrid_search(
        query=args.query,
        model=model,
        table=table,
        bm25_retriever=bm25_retriever,
        top_k=args.top_k,
        candidate_k=max(args.top_k * 4, 20),
        category=args.category,
    )

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Article ID:", result["article_id"])
        print("Category:", result["category"])
        print("RRF Score:", round(result["rrf_score"], 6))
        print("Text:", result["chunk"][:300])
        print("-" * 80)


if __name__ == "__main__":
    main()

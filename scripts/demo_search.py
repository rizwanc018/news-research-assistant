from news_research.embeddings import load_embedding_model
from news_research.retrieval.bm25_retrieval import bm25_search, load_bm25_index
from news_research.retrieval.hybrid_retrieval import hybrid_search
from news_research.retrieval.semantic_retrieval import semantic_search
from news_research.vector_store import connect_to_lancedb


def main() -> None:
    db = connect_to_lancedb()
    table = db.open_table("news_chunks")
    model = load_embedding_model()
    bm25_retriever = load_bm25_index()

    query = "What did Tony Blair say about the election?"

    print("\n=== SEMANTIC SEARCH ===")
    semantic_results = semantic_search(query=query, model=model, table=table, top_k=5)
    for index, result in semantic_results.iterrows():
        print(f"\nResult {index + 1}")
        print("Article ID:", result["article_id"])
        print("Category:", result["category"])
        print("Cosine Similarity:", 1 - result["_distance"])
        print("Text:", result["chunk"][:300])
        print("-" * 80)

    print("\n=== BM25 SEARCH ===")
    bm25_results = bm25_search(query=query, retriever=bm25_retriever, top_k=5)
    for i, result in enumerate(bm25_results, start=1):
        print(f"\nResult {i}")
        print("Article ID:", result["article_id"])
        print("Category:", result["category"])
        print("BM25 Score:", result["bm25_score"])
        print("Text:", result["chunk"][:300])
        print("-" * 80)

    print("\n=== HYBRID SEARCH (RRF) ===")
    hybrid_results = hybrid_search(
        query=query,
        model=model,
        table=table,
        bm25_retriever=bm25_retriever,
        top_k=5,
        candidate_k=20,
    )
    for i, result in enumerate(hybrid_results, start=1):
        print(f"\nResult {i}")
        print("Article ID:", result["article_id"])
        print("Category:", result["category"])
        print("RRF Score:", round(result["rrf_score"], 6))
        print("Text:", result["chunk"][:300])
        print("-" * 80)


if __name__ == "__main__":
    main()

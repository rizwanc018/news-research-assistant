from news_research.embeddings import load_embedding_model
from news_research.retrieval.bm25_retrieval import load_bm25_index
from news_research.retrieval.hybrid_retrieval import hybrid_search
from news_research.vector_store import connect_to_lancedb

db = connect_to_lancedb()
table = db.open_table("news_chunks")

model = load_embedding_model()
bm25_retriever = load_bm25_index()

query = "What did Tony Blair say about the election?"

results = hybrid_search(
    query=query,
    model=model,
    table=table,
    bm25_retriever=bm25_retriever,
    top_k=10,
    candidate_k=20,
    category="politics",
)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Chunk ID:", result["chunk_id"])
    print("Article ID:", result["article_id"])
    print("Category:", result["category"])
    print("RRF Score:", round(result["rrf_score"], 6))
    print("Semantic Rank:", result["semantic_rank"])
    print("BM25 Rank:", result["bm25_rank"])
    print("Text:", result["chunk"][:300])
    print("-" * 80)

assert all(result["category"] == "politics" for result in results)

print("Hybrid metadata filtering works!")

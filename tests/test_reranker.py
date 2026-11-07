from news_research.embeddings import load_embedding_model
from news_research.reranker import load_reranker, rerank_results
from news_research.retrieval.bm25_retrieval import load_bm25_index
from news_research.retrieval.retrieval_pipeline import retrieve_news
from news_research.vector_store import connect_to_lancedb

db = connect_to_lancedb()
table = db.open_table("news_chunks")

model = load_embedding_model()
bm25_retriever = load_bm25_index()

reranker = load_reranker()

query = "What did Tony Blair say about the election?"


response = retrieve_news(
    query=query,
    model=model,
    table=table,
    bm25_retriever=bm25_retriever,
)

# Use your existing hybrid retrieval results
reranked = rerank_results(
    query=query,
    results=response["results"],
    reranker=reranker,
    top_k=5,
)

for rank, result in enumerate(reranked, start=1):
    print(f"\nRank: {rank}")
    print("Chunk ID:", result["chunk_id"])
    print("RRF Score:", result["rrf_score"])
    print("Rerank Score:", result["rerank_score"])
    print("Chunk:", result["chunk"][:200])

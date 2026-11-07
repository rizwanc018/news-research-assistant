from news_research.embeddings import load_embedding_model
from news_research.reranker import load_reranker
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
    reranker=reranker,
)

print("Parsed query:")
print(response["parsed_query"].model_dump())

print("\nFinal results:")

for rank, result in enumerate(response["results"], start=1):
    print(f"\nRank: {rank}")
    print("Chunk ID:", result["chunk_id"])
    print("RRF score:", result["rrf_score"])
    print("Rerank score:", result["rerank_score"])
    print("Chunk:", result["chunk"][:200])

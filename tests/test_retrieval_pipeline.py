from news_research.embeddings import load_embedding_model
from news_research.retrieval.bm25_retrieval import load_bm25_index
from news_research.retrieval.retrieval_pipeline import retrieve_news
from news_research.vector_store import connect_to_lancedb

db = connect_to_lancedb()
table = db.open_table("news_chunks")

model = load_embedding_model()
bm25_retriever = load_bm25_index()

response = retrieve_news(
    query="Find technology news about artificial intelligence.",
    model=model,
    table=table,
    bm25_retriever=bm25_retriever,
)

print("Parsed query:")
print(response["parsed_query"].model_dump())

print("\nRetrieved chunks:")
for result in response["results"]:
    print(result["chunk_id"], result["rrf_score"])
    print(result["chunk"])
    print("_" * 50)

from news_research.retrieval.bm25_retrieval import (
    get_bm25_retriever,
    load_bm25_documents,
)
from news_research.vector_store import connect_to_lancedb

db = connect_to_lancedb()
table = db.open_table("news_chunks")

documents = load_bm25_documents(table)

bm25_retriever = get_bm25_retriever(documents)

print("Total documents:", len(documents))
print("BM25 index created successfully!")

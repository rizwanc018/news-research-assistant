from networkx import display

from news_reasearch.bm25_retrieval import bm25_search, load_bm25_index
from news_reasearch.embeddings import load_embedding_model
from news_reasearch.semantic_retrieval import semantic_search
from news_reasearch.vector_store import connect_to_lancedb

db = connect_to_lancedb()
table = db.open_table("news_chunks")
model = load_embedding_model()

bm25_retriever = load_bm25_index()


query = "What did Tony Blair say about the election?"


results = semantic_search(
    query=query,
    model=model,
    table=table,
    top_k=5,
)

print("\nSEMANTIC_SEARCH_RESULT")
display(results)
# for index, result in results.iterrows():
#     print(f"\nResult {index + 1}")
#     print("Article ID:", result["article_id"])
#     print("Category:", result["category"])
#     print("Cosine Distance:", result["_distance"])
#     print("Cosine Similarity:", 1 - result["_distance"])
#     print("Text:", result["chunk"])
#     print("-" * 80)


results = bm25_search(
    query=query,
    retriever=bm25_retriever,
    top_k=5,
)

print("\nBM25_SEARCH_RESULT")
for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Article ID:", result["article_id"])
    print("Category:", result["category"])
    print("BM25 Score:", result["bm25_score"])
    print("Text:", result["chunk"])
    print("-" * 80)

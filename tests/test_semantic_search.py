from news_research.embeddings import load_embedding_model
from news_research.retrieval.semantic_retrieval import semantic_search
from news_research.vector_store import connect_to_lancedb

model = load_embedding_model()

db = connect_to_lancedb()
table = db.open_table("news_chunks")

query = "How does consumer debt affect financial stability?"

results = semantic_search(
    query=query,
    model=model,
    table=table,
    top_k=5,
    category="politics",
)

print(results[["article_id", "category", "_distance"]])

for index, result in results.iterrows():
    print(f"\nResult {index + 1}")
    print("Article ID:", result["article_id"])
    print("Category:", result["category"])
    print("Cosine Distance:", result["_distance"])
    print("Cosine Similarity:", 1 - result["_distance"])
    print("Text:", result["chunk"])
    print("-" * 80)

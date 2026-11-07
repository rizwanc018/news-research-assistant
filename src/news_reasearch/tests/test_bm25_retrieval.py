from news_reasearch.retrieval.bm25_retrieval import bm25_search, load_bm25_index

bm25_retriever = load_bm25_index()

query = "What did Tony Blair say about the election?"

results = bm25_search(
    query=query,
    retriever=bm25_retriever,
    top_k=5,
    category="politics",
)

print("\n BM25_RESULT")
for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Article ID:", result["article_id"])
    print("Category:", result["category"])
    print("BM25 Score:", result["bm25_score"])
    # print("Text:", result["chunk"])
    # print("-" * 80)

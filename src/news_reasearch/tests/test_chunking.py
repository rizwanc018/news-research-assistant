from pathlib import Path

from news_reasearch.chunking import chunk_articles
from news_reasearch.preprocessing import load_articles, prepare_articles

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "bbc-news.csv"

df = load_articles(DATASET_PATH)
articles = prepare_articles(df)

chunks = chunk_articles(articles)

print("Total articles:", len(articles))
print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0])

print("\nSecond chunk:")
print(chunks[1])

print("\nChunks per article:")
article_id = articles[0]["article_id"]

for chunk in chunks:
    if chunk["article_id"] == article_id:
        print(chunk["chunk_id"], len(chunk["chunk"]))


chunk_ids = [chunk["chunk_id"] for chunk in chunks]

print("\nTotal chunks:", len(chunks))
print("Unique chunk IDs:", len(set(chunk_ids)))

assert len(chunk_ids) == len(set(chunk_ids))
assert all(chunk["chunk"] for chunk in chunks)
assert all(len(chunk["chunk"]) <= 500 for chunk in chunks)


for size, overlap in [(300, 50), (500, 75), (800, 100)]:
    chunks = chunk_articles(
        articles,
        chunk_size=size,
        overlap=overlap,
    )

    print(f"Size: {size}, Overlap: {overlap}, Total chunks: {len(chunks)}")

from news_reasearch.chunking import chunk_articles
from news_reasearch.embeddings import (
    embed_chunks,
    load_embedding_model,
)
from news_reasearch.preprocessing import load_articles, prepare_articles
from news_reasearch.vector_store import store_chunks

# Load and preprocess
df = load_articles("data/raw/bbc-news.csv")
articles = prepare_articles(df)

# Generate chunks
chunks = chunk_articles(
    articles,
    chunk_size=500,
    overlap=75,
)

# Load embedding model

model = load_embedding_model()

embedded_chunks = embed_chunks(
    model=model,
    chunks=chunks,
    batch_size=32,
)

table = store_chunks(embedded_chunks)

print("Total records stored:", table.count_rows())

print("\nFirst stored record:")
print(table.to_pandas().head(1))

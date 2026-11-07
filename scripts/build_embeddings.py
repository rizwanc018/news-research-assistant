from news_research.chunking import chunk_articles
from news_research.config import DATASET_PATH
from news_research.embeddings import (
    embed_chunks,
    load_embedding_model,
)
from news_research.preprocessing import load_articles, prepare_articles
from news_research.vector_store import store_chunks

# Load and preprocess
df = load_articles(DATASET_PATH)
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

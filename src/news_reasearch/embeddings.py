import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(
    model: SentenceTransformer, texts: list[str], batch_size: int = 32
) -> np.ndarray:
    if not texts:
        return np.empty((0, model.get_embedding_dimension()), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[dict],
    batch_size: int = 32,
) -> list[dict]:
    """Generate embeddings and attach them to their corresponding chunks."""

    texts = [chunk["chunk"] for chunk in chunks]

    embeddings = generate_embeddings(
        model=model,
        texts=texts,
        batch_size=batch_size,
    )

    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings do not match.")

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            {
                **chunk,
                "vector": embedding.tolist(),
            }
        )

    return embedded_chunks

from lancedb import Table
from sentence_transformers import SentenceTransformer


def semantic_search(
    query: str, model: SentenceTransformer, table: Table, top_k: int = 5
):
    query_embedding = model.encode(
        query, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )

    results = (
        table.search(query_embedding)
        .distance_type("cosine")
        .limit(top_k)
        .select(
            [
                "chunk_id",
                "article_id",
                "category",
                "chunk",
                "_distance",
            ]
        )
        .to_pandas()
    )

    return results

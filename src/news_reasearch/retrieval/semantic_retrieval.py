from lancedb import Table
from sentence_transformers import SentenceTransformer


def semantic_search(
    query: str,
    model: SentenceTransformer,
    table: Table,
    top_k: int = 5,
    category: str | None = None,
):
    query_embedding = model.encode(
        query, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )

    search_query = table.search(query_embedding).distance_type("cosine")

    if category is not None:
        search_query = search_query.where(
            f"category = '{category}'",
            prefilter=True,
        )

    results = (
        search_query.limit(top_k)
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

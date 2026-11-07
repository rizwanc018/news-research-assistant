from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker() -> CrossEncoder:
    """Load the Cross-Encoder model."""
    return CrossEncoder(MODEL_NAME)


def rerank_results(
    query: str, results: list[dict], reranker: CrossEncoder, top_k: int = 5
) -> list[dict]:
    if not results:
        return []

    pairs = [(query, result["chunk"]) for result in results]

    scores = reranker.predict(pairs)

    reranked_results = []
    for result, score in zip(results, scores):
        reranked_results.append({**result, "rerank_score": float(score)})

    reranked_results.sort(key=lambda item: item["rerank_score"], reverse=True)

    return reranked_results

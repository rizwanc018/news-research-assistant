def split_text(text: str, chunk_size: int = 500, overlap: int = 75) -> list[str]:
    """Split text into overlapping fixed-size chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be between 0 and chunk_size.")

    chunks = []

    step_size = chunk_size - overlap

    for start in range(0, len(text), step_size):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


def chunk_articles(
    articles: list[dict], chunk_size: int = 500, overlap: int = 75
) -> list[dict]:
    """Split articles into chunks while presrving metadata."""

    all_chunks = []

    for article in articles:
        article_chunks = split_text(
            article["text"], chunk_size=chunk_size, overlap=overlap
        )

        for i, chunk in enumerate(article_chunks):
            chunk_record = {
                "chunk_id": f"{article['article_id']}-chunk-{i:03d}",
                "article_id": article["article_id"],
                "category": article["category"],
                "chunk": chunk,
            }

            all_chunks.append(chunk_record)

    return all_chunks

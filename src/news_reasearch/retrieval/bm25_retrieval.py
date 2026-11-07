from pathlib import Path

import bm25s

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BM25_INDEX_PATH = PROJECT_ROOT / "data" / "bm25_index"


def load_bm25_documents(table) -> list[dict]:
    """Load news chunks and metadata from LanceDB."""

    df = table.to_pandas()

    documents = df[["chunk_id", "article_id", "category", "chunk"]].to_dict(
        orient="records"
    )

    return documents


def tokenize_documents(documents: list[dict]):
    texts = [document["chunk"] for document in documents]

    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=True)

    return corpus_tokens


def build_bm25_index(corpus_tokens, documents):
    """Build a BM25 index from tokenized documents."""

    retriever = bm25s.BM25(corpus=documents)

    retriever.index(corpus_tokens)

    return retriever


def save_bm25_index(retriever):
    """Save the BM25 index to disk."""
    retriever.save(BM25_INDEX_PATH)


def load_bm25_index():
    """Load an existing BM25 index from disk."""

    return bm25s.BM25.load(
        BM25_INDEX_PATH,
        load_corpus=True,
    )


def get_bm25_retriever(documents: list[dict]):
    """Load an existing BM25 index or build a new one."""

    index_path = Path(BM25_INDEX_PATH)

    if index_path.exists():
        print("Loading existing BM25 index...")
        return load_bm25_index()

    print("Building BM25 index...")

    corpus_tokens = tokenize_documents(documents)

    retriever = build_bm25_index(corpus_tokens, documents)

    save_bm25_index(retriever)


def bm25_search(
    query: str,
    retriever,
    top_k: int = 5,
    category: str | None = None,
) -> list[dict]:
    """Retrieve news chunks using BM25."""

    query_tokens = bm25s.tokenize(
        query,
        stopwords="en",
    )

    matched_documents, scores = retriever.retrieve(
        query_tokens,
        k=top_k,
    )

    results = []

    for document, score in zip(
        matched_documents[0],
        scores[0],
    ):
        if category is not None and document["category"] != category:
            continue

        results.append(
            {
                **document,
                "bm25_score": float(score),
            }
        )

        if len(results) == top_k:
            break

    return results

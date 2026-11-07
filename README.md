# news-research

A small RAG retrieval playground over the BBC news dataset. It builds two
indexes over article chunks — dense embeddings (LanceDB) and sparse BM25 — and
combines them with Reciprocal Rank Fusion (RRF) for hybrid search.

## Layout

```
src/news_research/        # the importable library
├── config.py              # project paths + shared constants
├── preprocessing.py       # load + clean articles
├── chunking.py            # split articles into overlapping chunks
├── embeddings.py          # sentence-transformer embeddings
├── vector_store.py        # LanceDB connection + writes
├── cli.py                 # `news-research` command
└── retrieval/
    ├── semantic_retrieval.py
    ├── bm25_retrieval.py
    └── hybrid_retrieval.py

scripts/                   # run-once pipeline + demos (not shipped)
├── build_embeddings.py    # chunk + embed + store in LanceDB
├── build_bm25_index.py    # build + save the BM25 index
├── inspect_data.py        # peek at the raw dataset
└── demo_search.py         # run all three search modes

tests/                     # runnable checks
data/raw/bbc-news.csv      # source dataset (generated indexes are gitignored)
```

## Setup

```bash
uv sync
```

## Build the indexes

```bash
uv run python scripts/build_embeddings.py   # -> data/lancedb/
uv run python scripts/build_bm25_index.py   # -> data/bm25_index/
```

## Search

```bash
uv run news-research "What did Tony Blair say about the election?" --top-k 5
# or run the full demo across semantic / BM25 / hybrid:
uv run python scripts/demo_search.py
```

All paths are resolved relative to the project root (see `config.py`), so
commands work from any directory.

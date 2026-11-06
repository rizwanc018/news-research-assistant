from pathlib import Path

import lancedb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "lancedb"
TABLE_NAME = "news_chunks"


def connect_to_lancedb():
    return lancedb.connect(DB_PATH)


def store_chunks(embedded_chunks: list[dict]):
    """Store embedded chunks in LanceDB."""

    if not embedded_chunks:
        raise ValueError("No embedded chunks to store.")

    db = connect_to_lancedb()

    table = db.create_table(
        TABLE_NAME,
        data=embedded_chunks,
        mode="overwrite",
    )

    return table

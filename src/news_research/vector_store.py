import lancedb

from news_research.config import LANCEDB_PATH, TABLE_NAME


def connect_to_lancedb():
    return lancedb.connect(LANCEDB_PATH)


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

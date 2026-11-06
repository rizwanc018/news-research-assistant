from news_reasearch.vector_store import connect_to_lancedb

db = connect_to_lancedb()

print(db.list_tables())

table = db.open_table("news_chunks")

print("Total rows:", table.count_rows())
print("Columns:", table.schema)

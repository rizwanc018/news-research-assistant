from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_PATH = RAW_DATA_DIR / "bbc-news.csv"

LANCEDB_PATH = DATA_DIR / "lancedb"
BM25_INDEX_PATH = DATA_DIR / "bm25_index"

TABLE_NAME = "news_chunks"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

client = OpenAI()

"""Quick look at the raw BBC news dataset.

    uv run python scripts/inspect_data.py
"""

import pandas as pd

from news_research.config import DATASET_PATH

df = pd.read_csv(DATASET_PATH)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

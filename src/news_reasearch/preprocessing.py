import json
from pathlib import Path

import pandas as pd


def load_articles(path: str) -> pd.DataFrame:
    """Load news articles from a CSV file."""
    return pd.read_csv(path)


def clean_article(article: dict) -> dict:
    article["category"] = article["category"].strip().lower()
    article["text"] = " ".join(article["text"].split())
    return article


def prepare_articles(dataframe: pd.DataFrame) -> list[dict]:
    df = dataframe.copy()

    df = df.dropna(subset=["category", "text"])

    df["text"] = df["text"].astype(str).str.split().str.join(" ")
    df["category"] = df["category"].astype(str).str.strip().str.lower()

    df = df[df["text"] != ""]

    df = df.drop_duplicates(subset=["text"], keep="first")

    df = df.reset_index(drop=True)
    df["article_id"] = [f"article-{i:04d}" for i in range(1, len(df) + 1)]

    return df.to_dict(orient="records")


def save_articles(articles: list[dict], path: str) -> None:
    """Save processed articles to JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(articles, file, indent=4, ensure_ascii=False)

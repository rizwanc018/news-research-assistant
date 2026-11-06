import pandas as pd

df = pd.read_csv("raw/bbc-news.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nCategories:")
print(df["category"].value_counts())

print("\nDuplicate rows:", df.duplicated().sum())

print("Duplicate article texts:", df["text"].duplicated().sum())

print("Empty article texts:", (df["text"].str.strip() == "").sum())

print("\nArticle length statistics:")
print(df["text"].str.len().describe())

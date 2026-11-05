from src.preprocessing import load_articles, prepare_articles

DATASET_PATH = "data/raw/bbc-news.csv"

df = load_articles(DATASET_PATH)

articles = prepare_articles(df)

print("Original articles:", len(df))
print("Clean articles:", len(articles))

print("\nFirst cleaned article:")
print(articles[0])

print("\nCategory distribution:")

for category in sorted({a["category"] for a in articles}):
    count = sum(a["category"] == category for a in articles)
    print(category, count)

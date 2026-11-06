import bm25s

text = "The government increased interest rates."

tokens = bm25s.tokenize(
    text,
    stopwords="en",
    return_ids=False,
)

print(tokens)

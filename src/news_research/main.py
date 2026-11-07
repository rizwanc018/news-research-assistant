from news_research.embeddings import load_embedding_model
from news_research.reranker import load_reranker
from news_research.retrieval.bm25_retrieval import load_bm25_index
from news_research.retrieval.retrieval_pipeline import retrieve_news
from news_research.vector_store import connect_to_lancedb


def main() -> None:
    db = connect_to_lancedb()
    table = db.open_table("news_chunks")

    model = load_embedding_model()
    bm25_retriever = load_bm25_index()

    reranker = load_reranker()

    query = "What did Tony Blair say about the election?"

    response = retrieve_news(
        query=query,
        model=model,
        table=table,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )

    print("\nParsed query:")
    print(response["parsed_query"].model_dump())

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")
    for result in response["results"]:
        print(result["article_id"], result["chunk_id"])


if __name__ == "__main__":
    main()

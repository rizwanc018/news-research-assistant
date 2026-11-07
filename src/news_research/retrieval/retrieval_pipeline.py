from news_research.query_parser import parse_query
from news_research.reranker import rerank_results
from news_research.retrieval.hybrid_retrieval import hybrid_search


def retrieve_news(
    query: str,
    model,
    table,
    bm25_retriever,
    reranker,
    top_k: int = 5,
    candidate_k: int = 20,
):
    print(">> Parsing querry...")
    parsed_query = parse_query(query)

    if parsed_query.intent == "out_of_scope":
        return {
            "parsed_query": parsed_query,
            "results": [],
        }

    print("Retriving...")
    hybrid_results = hybrid_search(
        query=parsed_query.search_query,
        model=model,
        table=table,
        bm25_retriever=bm25_retriever,
        top_k=candidate_k,
        candidate_k=candidate_k,
        category=parsed_query.category,
    )

    print("Reranking...")
    reranked_results = rerank_results(
        query=query,
        results=hybrid_results,
        reranker=reranker,
        top_k=top_k,
    )

    return {
        "parsed_query": parsed_query,
        "results": reranked_results,
    }

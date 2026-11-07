from news_research.generation import generate_answer
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
    parsed_query = parse_query(query)

    if parsed_query.intent == "out_of_scope":
        return {
            "parsed_query": parsed_query,
            "results": [],
        }

    hybrid_results = hybrid_search(
        query=parsed_query.search_query,
        model=model,
        table=table,
        bm25_retriever=bm25_retriever,
        top_k=candidate_k,
        candidate_k=candidate_k,
        category=parsed_query.category,
    )

    reranked_results = rerank_results(
        query=query,
        results=hybrid_results,
        reranker=reranker,
        top_k=top_k,
    )

    answer = generate_answer(
        query=query,
        results=reranked_results,
    )

    return {
        "parsed_query": parsed_query,
        "results": reranked_results,
        "answer": answer,
    }

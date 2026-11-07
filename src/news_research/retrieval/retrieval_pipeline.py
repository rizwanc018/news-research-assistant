from news_research.query_parser import parse_query
from news_research.retrieval.hybrid_retrieval import hybrid_search


def retrieve_news(
    query: str,
    model,
    table,
    bm25_retriever,
    top_k: int = 10,
    candidate_k: int = 20,
):
    # Step 1: Understand the user's query
    parsed_query = parse_query(query)

    # Step 2: Skip retrieval for unrelated questions
    if parsed_query.intent == "out_of_scope":
        return {
            "parsed_query": parsed_query,
            "results": [],
        }

    # Step 3: Retrieve relevant news chunks
    results = hybrid_search(
        query=parsed_query.search_query,
        model=model,
        table=table,
        bm25_retriever=bm25_retriever,
        top_k=top_k,
        candidate_k=candidate_k,
        category=parsed_query.category,
    )

    return {
        "parsed_query": parsed_query,
        "results": results,
    }

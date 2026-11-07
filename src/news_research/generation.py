from news_research.config import MODEL_NAME, client


def format_context(results: list[dict]) -> str:

    context_parts = []

    for result in results:
        context_parts.append(
            f"""
Article ID: {result["article_id"]}
Category: {result["category"]}
Content: {result["chunk"]}
"""
        )

    return "\n".join(context_parts)


def generate_answer(
    query: str,
    results: list[dict],
) -> str:

    if not results:
        return "I couldn't find relevant news articles to answer your question."

    context = format_context(results)

    response = client.responses.create(
        model=MODEL_NAME,
        instructions="""
        You are a helpful news research assistant.

        Answer the user's question using only the provided
        news article context.

        Rules:
        1. Do not use outside knowledge.
        2. Do not invent facts, dates, events, or statements.
        3. If the context does not contain enough information,
           clearly say that.
        4. Provide a clear and concise answer.
        5. When possible, mention the article IDs that support
           your answer.
        """,
        input=f"""
        News article context:

        {context}

        User question:

        {query}
        """,
    )

    return response.output_text

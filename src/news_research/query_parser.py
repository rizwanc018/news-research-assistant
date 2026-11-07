from typing import Literal

from pydantic import BaseModel, Field

from news_research.config import client


class ParsedQuery(BaseModel):
    intent: Literal["search", "question", "out_of_scope"] = Field(
        description="The user's request type."
    )
    search_query: str = Field(
        description="The actual search query used for news retrieval."
    )
    category: (
        Literal["business", "entertainment", "politics", "sport", "tech"] | None
    ) = Field(
        default=None, description="News category filter, or null if none is specified."
    )


def parse_query(query: str) -> ParsedQuery:
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": """
                You are a query parser for a news research assistant.

                Your task is to understand the user's request and
                convert it into structured information.

                Rules:
                1. Identify whether the user wants to search for news,
                   ask a question about news, or asks something
                   unrelated to news.

                2. Use "search" when the user wants to find or browse
                   news articles.

                3. Use "question" when the user asks a question
                   that can be answered using news articles.

                4. Use "out_of_scope" when the request is unrelated
                   to news research.

                5. Extract a concise search query containing the
                   important people, events, topics, and keywords.

                6. Extract a category only when the user explicitly
                   specifies or clearly requests one.

                7. If no category is specified, return null.

                Available categories:
                business, entertainment, politics, sport, tech.
                """,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        text_format=ParsedQuery,
    )

    return response.output_parsed

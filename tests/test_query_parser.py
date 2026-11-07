from news_research.query_parser import parse_query

query = "Find political news about Tony Blair and the election."

parsed = parse_query(query)

print(parsed.model_dump())

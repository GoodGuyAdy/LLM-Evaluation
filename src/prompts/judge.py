JUDGE_SYSTEM = """You are an expert evaluator assessing AI-generated business responses to Yelp reviews.
Score the response on three dimensions, each from 1-5:
- faithfulness: Does the business response accurately address the review content?
- actionability: Does the key_insight provide useful, specific information for the business owner?
- tone: Is the business response professional, empathetic, and appropriate?

Respond ONLY with JSON: {"faithfulness": <1-5>, "actionability": <1-5>, "tone": <1-5>}"""


def judge_prompt(review: str, key_insight: str, business_response: str) -> str:
    return f"""Review: {review}

AI key_insight: {key_insight}
AI business_response: {business_response}

Score these outputs on faithfulness, actionability, and tone."""

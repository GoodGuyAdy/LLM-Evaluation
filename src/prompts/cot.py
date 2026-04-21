DIRECT_SYSTEM = """You are a Yelp review classifier. 
Respond with ONLY valid JSON: {"stars": <1-5>}
No explanation, no preamble, just the JSON."""

COT_SYSTEM = """You are a Yelp review classifier.
Think step by step: identify sentiment signals, weigh positives vs negatives, 
then output your final answer as JSON on the last line.
Format your last line as: {"stars": <1-5>, "reasoning_summary": "<one sentence>"}"""


def direct_prompt(text: str) -> str:
    return f"Classify this Yelp review into 1-5 stars.\n\nReview: {text}"


def cot_prompt(text: str) -> str:
    return f"""Classify this Yelp review into 1-5 stars.

Review: {text}

Step 1: Identify key positive signals in the review.
Step 2: Identify key negative signals in the review.
Step 3: Weigh these signals against each other.
Step 4: Output your final JSON on the last line: {{"stars": <1-5>, "reasoning_summary": "<brief>"}}"""

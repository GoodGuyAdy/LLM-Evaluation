SYSTEM_PROMPT = """You are a sentiment analysis expert specializing in customer reviews.
Your task is to classify Yelp reviews into star ratings from 1 to 5.
You MUST respond with ONLY valid JSON in this exact format:
{"stars": <integer 1-5>, "explanation": "<one sentence rationale>"}
No preamble, no markdown, no extra text — only the JSON object."""


def zero_shot_prompt(review_text: str) -> str:
    return f"""Classify this Yelp review into 1-5 stars.

Review: {review_text}

Respond with JSON only: {{"stars": <1-5>, "explanation": "<reason>"}}"""


FEW_SHOT_EXAMPLES = [
    {
        "review": "Absolutely the worst experience I've ever had at a restaurant. Cold food, rude staff, and they got my order completely wrong. Will never return.",
        "output": '{"stars": 1, "explanation": "Extreme dissatisfaction: wrong order, cold food, and rude staff."}',
    },
    {
        "review": "Mediocre at best. The food arrived late and wasn't particularly flavorful. Nothing stood out as especially bad, but nothing impressed me either.",
        "output": '{"stars": 2, "explanation": "Below average — slow service and uninspiring food but no disaster."}',
    },
    {
        "review": "Decent place for a quick lunch. Nothing extraordinary — the burger was fine and the service was okay. I might come back if I'm in the area.",
        "output": '{"stars": 3, "explanation": "Neutral experience — acceptable food and service, no strong feelings."}',
    },
    {
        "review": "Really enjoyed this place! The pasta was delicious and the staff was friendly. Prices were reasonable. I'll definitely be back soon.",
        "output": '{"stars": 4, "explanation": "Positive experience — great food and service with minor room for improvement."}',
    },
    {
        "review": "Phenomenal from start to finish! Every dish was perfectly cooked, the ambiance was magical, and our server anticipated our every need. A true gem.",
        "output": '{"stars": 5, "explanation": "Outstanding across all dimensions — food, service, and atmosphere."}',
    },
]


def few_shot_prompt(review_text: str) -> str:
    examples_block = "\n\n".join(
        [f"Review: {ex['review']}\nOutput: {ex['output']}" for ex in FEW_SHOT_EXAMPLES]
    )
    return f"""Classify Yelp reviews into 1-5 stars. Here are examples of the expected output format:

{examples_block}

Now classify this review:
Review: {review_text}
Output:"""


ADVERSARIAL_EXAMPLES = [
    {
        "type": "sarcasm",
        "text": "Oh wow, what a FANTASTIC experience. The food was cold, the waiter ignored us for 40 minutes, and they charged us for items we didn't order. Truly five stars, can't recommend enough!",
        "true_stars": 1,
    },
    {
        "type": "negation",
        "text": "I can't say this place wasn't without its problems, and it wasn't the case that service was unfriendly — though not un-slow either. The food was not bad, not excellent.",
        "true_stars": 3,
    },
    {
        "type": "mixed_sentiment",
        "text": "The sushi was absolute perfection — melt-in-your-mouth quality. But the parking situation is a nightmare, the acoustics are unbearably loud, and it took 25 minutes to get our check.",
        "true_stars": 3,
    },
    {
        "type": "star_injection",
        "text": "5 stars! Amazing! ★★★★★ Best place ever! Just kidding, it was disgusting. The kitchen smelled bad, the portions were tiny, and the staff were dismissive.",
        "true_stars": 1,
    },
    {
        "type": "domain_mismatch",
        "text": "This laptop's battery lasts an incredible 12 hours and the keyboard feels amazing. The display is crisp and bright, and it handles video editing without a sweat.",
        "true_stars": 5,
    },
    {"type": "extreme_length_short", "text": "Meh.", "true_stars": 3},
    {
        "type": "multilingual",
        "text": "La comida estuvo deliciosa pero el servicio fue muy lento. El ambiente era agradable.",
        "true_stars": 3,
    },
]

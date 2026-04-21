import time
import pandas as pd
from tqdm import tqdm
from src.models.llm_client import OpenAIClient
from src.prompts.assistant import assistant_prompt, ASSISTANT_SYSTEM
from src.prompts.judge import judge_prompt, JUDGE_SYSTEM
from datasets import load_dataset
from src.evaluation.parser import parse_json_response
from src.utils.helpers import save_results


def run():
    client = OpenAIClient()

    yelp_dataset = load_dataset("yelp_review_full", split="test")

    ASSIST_SAMPLE = 100

    df_full = pd.DataFrame(
        {
            "text": yelp_dataset["text"],
            "label": [int(l) + 1 for l in yelp_dataset["label"]],
        }
    )

    df_assist_eval = (
        df_full.groupby("label")
        .sample(n=ASSIST_SAMPLE // 5, random_state=99)
        .reset_index(drop=True)
    )

    assist_results = []
    for _, row in tqdm(
        df_assist_eval.iterrows(), total=len(df_assist_eval), desc="Multi-task"
    ):
        raw = client.generate(ASSISTANT_SYSTEM, assistant_prompt(row["text"]))
        parsed = parse_json_response(raw)
        assist_results.append(
            {
                "true_label": row["label"],
                "review_snippet": row["text"][:150],
                "pred_stars": parsed.get("stars") if parsed else None,
                "key_insight": parsed.get("key_insight") if parsed else None,
                "business_response": parsed.get("business_response")
                if parsed
                else None,
                "valid_json": parsed is not None,
                "raw": raw,
            }
        )
        time.sleep(0.3)

    df_assist = pd.DataFrame(assist_results)

    print(f"Multi-task complete. JSON compliance: {df_assist['valid_json'].mean():.1%}")

    judge_sample = df_assist.dropna(subset=["key_insight", "business_response"]).sample(
        30, random_state=42
    )
    judge_scores = []

    for _, row in tqdm(
        judge_sample.iterrows(), total=len(judge_sample), desc="LLM-judge"
    ):
        raw = client.generate(
            JUDGE_SYSTEM,
            judge_prompt(
                row["review_snippet"], row["key_insight"], row["business_response"]
            ),
        )
        parsed = parse_json_response(raw)
        if parsed:
            judge_scores.append(
                {
                    "faithfulness": parsed.get("faithfulness"),
                    "actionability": parsed.get("actionability"),
                    "tone": parsed.get("tone"),
                }
            )
        time.sleep(0.3)

    df_judge = pd.DataFrame(judge_scores)

    print("LLM-as-Judge Scores (mean, out of 5)")
    print(df_judge.mean().round(2))

    print("Example Multi-Task Outputs")
    for star in [1, 3, 5]:
        row = (
            df_assist[df_assist["true_label"] == star]
            .dropna(subset=["key_insight"])
            .iloc[0]
        )

        print(f"TRUE STARS: {star} | PREDICTED: {row['pred_stars']}")
        print(f"REVIEW: {row['review_snippet']}...")
        print(f"KEY INSIGHT: {row['key_insight']}")
        print(f"BUSINESS RESPONSE: {row['business_response']}")

    save_results(df_assist, "task3_multitask_outputs")
    save_results(df_judge, "task3_llm_judge_scores")

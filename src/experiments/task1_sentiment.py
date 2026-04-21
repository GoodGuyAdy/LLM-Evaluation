import time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from src.models.llm_client import OpenAIClient
from src.prompts.sentiment import SYSTEM_PROMPT, zero_shot_prompt, few_shot_prompt
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.parser import parse_json_response
from src.utils.helpers import save_results


def run():
    client = OpenAIClient()

    yelp_dataset = load_dataset("yelp_review_full", split="test")

    EVAL_SAMPLE = 100

    df_full = pd.DataFrame(
        {
            "text": yelp_dataset["text"],
            "label": [int(l) + 1 for l in yelp_dataset["label"]],
        }
    )

    df_eval = df_full.groupby("label").sample(n=EVAL_SAMPLE // 5).reset_index(drop=True)

    zero_shot_results = []
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Zero-shot"):
        raw = client.generate(SYSTEM_PROMPT, zero_shot_prompt(row["text"]))
        parsed = parse_json_response(raw)
        pred = parsed.get("stars") if parsed else None
        explanation = parsed.get("explanation") if parsed else None
        zero_shot_results.append(
            {
                "true_label": row["label"],
                "pred_stars": pred,
                "explanation": explanation,
                "raw_response": raw,
                "valid_json": parsed is not None,
            }
        )
        time.sleep(0.3)

    df_zs = pd.DataFrame(zero_shot_results)
    print("Zero-shot complete.")

    few_shot_results = []
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Few-shot"):
        raw = client.generate(SYSTEM_PROMPT, few_shot_prompt(row["text"]))
        parsed = parse_json_response(raw)
        pred = parsed.get("stars") if parsed else None
        explanation = parsed.get("explanation") if parsed else None
        few_shot_results.append(
            {
                "true_label": row["label"],
                "pred_stars": pred,
                "explanation": explanation,
                "raw_response": raw,
                "valid_json": parsed is not None,
            }
        )
        time.sleep(0.3)

    df_fs = pd.DataFrame(few_shot_results)
    print("Few-shot complete.")

    zs_metrics = evaluate_predictions(df_zs, "pred_stars", "true_label")
    fs_metrics = evaluate_predictions(df_fs, "pred_stars", "true_label")

    results_t1 = pd.DataFrame(
        [
            {"Strategy": "Zero-Shot", **zs_metrics},
            {"Strategy": "Few-Shot", **fs_metrics},
        ]
    )
    print("Task 1 Results:")
    print(results_t1.to_string(index=False))
    save_results(results_t1, "task1_zero_vs_few")

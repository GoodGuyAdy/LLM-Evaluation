import time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from src.models.llm_client import OpenAIClient
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.parser import parse_json_response
from src.prompts.cot import direct_prompt, DIRECT_SYSTEM, cot_prompt, COT_SYSTEM
from src.utils.helpers import save_results


def run():
    client = OpenAIClient()

    yelp_dataset = load_dataset("yelp_review_full", split="test")

    EVAL_SAMPLE = 200

    df_full = pd.DataFrame(
        {
            "text": yelp_dataset["text"],
            "label": [int(l) + 1 for l in yelp_dataset["label"]],
        }
    )

    df_eval = df_full.groupby("label").sample(n=EVAL_SAMPLE // 5).reset_index(drop=True)
    direct_results, cot_results = [], []

    print("Running Direct vs CoT comparison...")
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        # Direct
        raw_d = client.generate(DIRECT_SYSTEM, direct_prompt(row["text"]))
        parsed_d = parse_json_response(raw_d)
        direct_results.append(
            {
                "true_label": row["label"],
                "pred_stars": parsed_d.get("stars") if parsed_d else None,
                "valid_json": parsed_d is not None,
                "raw": raw_d,
            }
        )

        # CoT
        raw_c = client.generate(COT_SYSTEM, cot_prompt(row["text"]))
        last_line = raw_c.strip().split("\n")[-1]
        parsed_c = parse_json_response(last_line) or parse_json_response(raw_c)
        cot_results.append(
            {
                "true_label": row["label"],
                "pred_stars": parsed_c.get("stars") if parsed_c else None,
                "reasoning_summary": parsed_c.get("reasoning_summary")
                if parsed_c
                else None,
                "valid_json": parsed_c is not None,
                "raw": raw_c,
            }
        )
        time.sleep(0.3)

    df_direct = pd.DataFrame(direct_results)
    df_cot = pd.DataFrame(cot_results)
    print("Done")

    dir_metrics = evaluate_predictions(df_direct, "pred_stars", "true_label")
    cot_metrics = evaluate_predictions(df_cot, "pred_stars", "true_label")

    results_t2 = pd.DataFrame(
        [
            {"Strategy": "Direct Answer", **dir_metrics},
            {"Strategy": "Chain-of-Thought", **cot_metrics},
        ]
    )
    print("Task 2 Results:")
    print(results_t2.to_string(index=False))

    # Error type analysis — classify errors by deviation magnitude
    def classify_error(true, pred):
        if pd.isna(pred):
            return "parse_failure"
        diff = abs(int(true) - int(pred))
        if diff == 0:
            return "correct"
        if diff == 1:
            return "off_by_1"
        if diff == 2:
            return "off_by_2"
        return "major_error"

    for df_r, name in [(df_direct, "Direct"), (df_cot, "CoT")]:
        df_r["error_type"] = df_r.apply(
            lambda r: classify_error(r["true_label"], r["pred_stars"]), axis=1
        )
        print(f"\n{name} error breakdown:")
        print(df_r["error_type"].value_counts())

    save_results(results_t2, "task2_direct_vs_cot")

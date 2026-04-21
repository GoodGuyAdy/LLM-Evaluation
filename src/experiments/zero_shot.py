import time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from src.models.llm_client import OpenAIClient
from src.prompts.sentiment import SYSTEM_PROMPT, zero_shot_prompt
from src.evaluation.parser import parse_json_response


def zero_shot_run():
    client = OpenAIClient()

    yelp_dataset = load_dataset("yelp_review_full", split="test")

    EVAL_SAMPLE = 25

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
    return zero_shot_results, df_zs

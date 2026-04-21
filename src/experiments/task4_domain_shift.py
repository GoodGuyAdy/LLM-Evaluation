import time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from src.models.llm_client import OpenAIClient
from src.evaluation.parser import parse_json_response
from src.prompts.sentiment import ADVERSARIAL_EXAMPLES
from src.prompts.domain import DOMAIN_AWARE_SYSTEM
from src.experiments.zero_shot import zero_shot_run
from src.prompts.sentiment import zero_shot_prompt
from src.utils.helpers import save_results


def run():
    client = OpenAIClient()

    print("Loading Amazon Polarity dataset...")
    amazon = load_dataset("amazon_polarity", split="test", trust_remote_code=True)
    df_amz = pd.DataFrame(amazon).sample(200, random_state=42)

    df_amz["text"] = (
        df_amz["content"] if "content" in df_amz.columns else df_amz["review_body"]
    )
    df_amz["true_label"] = df_amz["label"].map({0: 1, 1: 5})

    print("Loading IMDB dataset...")
    imdb = load_dataset("imdb", split="test")
    df_imdb = pd.DataFrame(imdb).sample(200, random_state=42)
    df_imdb["true_label"] = df_imdb["label"].map({0: 1, 1: 5})
    df_imdb.rename(columns={"text": "text"}, inplace=True)

    print(f"Amazon sample: {len(df_amz)}, IMDB sample: {len(df_imdb)}")

    # Evaluate on Amazon & IMDB (zero-shot, same prompt)

    def run_domain_eval(df_source, source_name, n=100):
        sample = df_source.sample(min(n, len(df_source)), random_state=42)
        results = []
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=source_name):
            raw = client.generate(None, zero_shot_prompt(row["text"][:600]))
            parsed = parse_json_response(raw)
            pred = parsed.get("stars") if parsed else None
            results.append(
                {
                    "true_label": row["true_label"],
                    "pred_stars": pred,
                    "valid_json": parsed is not None,
                }
            )
            time.sleep(0.3)
        return pd.DataFrame(results)

    df_amz_res = run_domain_eval(df_amz, "Amazon")
    df_imdb_res = run_domain_eval(df_imdb, "IMDB")

    adv_results = []
    for ex in tqdm(ADVERSARIAL_EXAMPLES, desc="Adversarial"):
        raw = client.generate(None, zero_shot_prompt(ex["text"]))
        parsed = parse_json_response(raw)
        pred = parsed.get("stars") if parsed else None
        adv_results.append(
            {**ex, "pred_stars": pred, "correct": pred == ex["true_stars"], "raw": raw}
        )
        time.sleep(0.3)

    df_adv = pd.DataFrame(adv_results)
    print("Adversarial Results:")
    print(
        df_adv[["type", "true_stars", "pred_stars", "correct"]].to_string(index=False)
    )

    # For Amazon/IMDB we only compare positive vs negative direction since labels are binary
    def binary_accuracy(df_r, threshold=3):
        valid = df_r.dropna(subset=["pred_stars"])
        true_pos = (valid["true_label"] >= threshold).astype(int)
        pred_pos = (valid["pred_stars"] >= threshold).astype(int)
        return accuracy_score(true_pos, pred_pos)

    _, df_zs = zero_shot_run()

    yelp_bin_acc = binary_accuracy(
        df_zs.rename(columns={"true_label": "true_label", "pred_stars": "pred_stars"})
    )
    amz_bin_acc = binary_accuracy(df_amz_res)
    imdb_bin_acc = binary_accuracy(df_imdb_res)

    domain_summary = pd.DataFrame(
        [
            {
                "Domain": "Yelp (in-domain)",
                "Binary Accuracy": yelp_bin_acc,
                "Parse Rate": df_zs["valid_json"].mean(),
            },
            {
                "Domain": "Amazon (out-domain)",
                "Binary Accuracy": amz_bin_acc,
                "Parse Rate": df_amz_res["valid_json"].mean(),
            },
            {
                "Domain": "IMDB (out-domain)",
                "Binary Accuracy": imdb_bin_acc,
                "Parse Rate": df_imdb_res["valid_json"].mean(),
            },
        ]
    )
    print("Domain Shift Summary:")
    print(domain_summary.to_string(index=False))

    mitigation_results = []
    imdb_sample = df_imdb.sample(30, random_state=77)
    for _, row in tqdm(imdb_sample.iterrows(), total=30, desc="Mitigation test"):
        raw = client.generate(DOMAIN_AWARE_SYSTEM, zero_shot_prompt(row["text"][:600]))
        parsed = parse_json_response(raw)
        pred = parsed.get("stars") if parsed else None
        mitigation_results.append({"true_label": row["true_label"], "pred_stars": pred})
        time.sleep(0.3)

    df_mit = pd.DataFrame(mitigation_results)
    mit_acc = binary_accuracy(df_mit)
    print(f"IMDB Accuracy BEFORE mitigation: {imdb_bin_acc:.2f}")
    print(f"IMDB Accuracy AFTER  mitigation: {mit_acc:.2f}")
    print(f"Improvement: {(mit_acc - imdb_bin_acc) * 100:+.1f} percentage points")

    save_results(domain_summary, "task4_domain_shift")
    save_results(df_adv, "task4_adversarial_results")

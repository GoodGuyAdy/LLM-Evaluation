import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def evaluate_predictions(
    df_results: pd.DataFrame, pred_col: str, true_col: str = "label"
):
    """Compute accuracy and macro-F1, excluding unparseable rows."""
    valid = df_results.dropna(subset=[pred_col])
    parse_rate = len(valid) / len(df_results)
    acc = accuracy_score(valid[true_col], valid[pred_col])
    f1 = f1_score(valid[true_col], valid[pred_col], average="macro", zero_division=0)
    return {
        "n_valid": len(valid),
        "parse_rate": parse_rate,
        "accuracy": acc,
        "macro_f1": f1,
    }

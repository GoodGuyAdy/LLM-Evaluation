import os
import json
from datetime import datetime
import pandas as pd


def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i : i + size]


def save_results(results, task_name: str):
    """
    Save results to JSON file.

    Args:
        results: pd.DataFrame OR dict OR list
        task_name (str): name of the task (e.g., 'task1_zero_vs_few')
        output_dir (str): folder to store outputs
    """
    output_dir = "outputs"

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Clean timestamp (safe for filenames)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # File name
    filename = f"{task_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert results to JSON serializable format
    if isinstance(results, pd.DataFrame):
        data = results.to_dict(orient="records")
    elif isinstance(results, dict):
        data = results
    elif isinstance(results, list):
        data = results
    else:
        raise ValueError("Unsupported result type")

    # Save JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved results to {filepath}")

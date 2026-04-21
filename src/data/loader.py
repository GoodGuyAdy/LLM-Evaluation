from datasets import load_dataset
import pandas as pd

def load_yelp():
    dataset = load_dataset("yelp_review_full")
    df = pd.DataFrame(dataset["train"])
    df.rename(columns={"label": "stars"}, inplace=True)
    df["stars"] += 1
    return df

def load_amazon():
    dataset = load_dataset("amazon_polarity")
    return pd.DataFrame(dataset["train"])

def load_imdb():
    dataset = load_dataset("imdb")
    return pd.DataFrame(dataset["train"])
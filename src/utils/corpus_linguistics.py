import spacy
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import argparse

artifacts_path = os.path.abspath("artifacts")

def load_data(filename):
    return pd.read_pickle(os.path.join(artifacts_path, filename))

def calculate_stats(df):
    nlp = spacy.load('en_core_web_sm')
    corpus_df = pd.DataFrame(df["name"], index=df.index)
    corpus_df["has_stopwords"] = np.nan
    corpus_df["lexical_density"] = np.nan
    corpus_df["has_digits"] = np.nan
    corpus_df["has_special_chars"] = np.nan
    print(f"{datetime.now()} - Computing corpus linguistics...")
    for i, r in tqdm(df.iterrows()):
        doc = nlp(r["name"])
        corpus_df.at[i, "has_stopwords"] = any(tok.is_stop for tok in doc)
        corpus_df.at[i, "lexical_density"] = round(compute_lexical_density(doc), 3)
        corpus_df.at[i, "has_digits"] = any(tok.text.isdigit() for tok in doc)
        corpus_df.at[i, "has_special_chars"] = any(not c.isalnum() for c in r["name"] if c != " ")
    corpus_df.to_pickle(os.path.join(artifacts_path, "corpus_stats.pkl"))
    print(f"{datetime.now()} - Finished!")
    return corpus_df

def compute_lexical_density(doc):
    open_class = {"VB", "JJ", "NN", "RB"}
    count = 0
    for tok in doc:
        if tok.tag_[:2] in open_class:
            count += 1
    return count / len(doc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    df = load_data(args.filename)
    corpus_df = calculate_stats(df)

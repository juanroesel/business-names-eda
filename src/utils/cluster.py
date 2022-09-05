from sklearn.cluster import OPTICS
import joblib
import pandas as pd
import os
from datetime import datetime
import argparse

artifacts_path = os.path.abspath("artifacts")

def load_vectors(filename):
    return pd.read_pickle(os.path.join(artifacts_path, filename))

def clustering_model(df):
    model = OPTICS(
        min_samples=5, 
        metric="euclidean", 
        n_jobs=-1, 
    ).fit_predict(df)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    vectors_df = load_vectors(args.filename)
    print(f"{datetime.now()} - Fitting OPTICS model against {len(vectors_df)} rows...")
    model = clustering_model(vectors_df)
    joblib.dump(model, os.path.join(artifacts_path, "optics.joblib"))
    print(f"{datetime.now()} - Finished!")

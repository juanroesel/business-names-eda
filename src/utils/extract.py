import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

artifacts_path = os.path.abspath("artifacts")

def load_data_and_sample(path=artifacts_path):
    df = pd.read_pickle(os.path.join(path, "feat_eng_df.pkl"))
    return df.sample(frac=0.02)


def partition_data(df, partitions=5):
    batches = np.array_split(df, partitions)
    for i, batch_df in enumerate(batches):
        batch_df.to_pickle(os.path.join(artifacts_path, f"batch_{i + 1}.pkl"))
    return batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions", type=int)
    args = parser.parse_args()
    print(f"{datetime.now()}Sampling and partitioning dataset in {args.partitions} partitions...")
    sampled_df = load_data_and_sample()
    _ = partition_data(sampled_df, partitions=args.partitions)
    print(f"{datetime.now()} - Finished!")
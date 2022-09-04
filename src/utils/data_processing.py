import pandas as pd
import numpy as np
import nlp_utils
import spacy
from datetime import datetime
from tqdm import tqdm
import os

data_path = os.path.abspath("data")
artifacts_path = os.path.abspath("artifacts")

def load_data(path=data_path):
    df = pd.read_csv(os.path.join(data_path, "companies_sorted.csv"))
    return df

def clean_dataset(df, outpath=artifacts_path):
    # remove duplicate names
    print(f"{datetime.now()} - Cleaning dataset... (original length = {len(df)})")
    df = df.drop_duplicates(subset=["name"])
    # remove column 'estimated company employees'
    df = df.drop("current employee estimate", axis=1)
    # remove rows with extreme outliers in 'year founded' field
    df["year founded"] = df["year founded"].apply(lambda x: int(x) if pd.notnull(x) else x)
    df = df.query("1850 <= `year founded` <= 2022")
    df.to_pickle(os.path.join(outpath, "cleaned_df.pkl"))
    print(f"{datetime.now()} - Finished! (new length = {len(df)})")
    return df


def feature_engineering(df, outpath=artifacts_path):
    # load spacy model
    nlp = spacy.load("en_core_web_sm")
    # create domain_type and lang fields
    df["domain_type"] = np.nan
    df["state"] = np.nan
    df["city"] = np.nan
    df["lang"] = np.nan
    print(f"{datetime.now()} - Starting feature engineering process...")
    for i, r in tqdm(df.iterrows()):
        # populate domain_type field
        df.at[i, "domain_type"] = r["domain"].split(".")[-1] if pd.notnull(r["domain"]) else r["domain_type"]
        # remove redundant portion of 'linkedin url' string
        df.at[i, "linkedin url"] = r["linkedin url"].split("/")[-1]
        # standardize 'locality' string
        df.at[i, "locality"] = r["locality"].lower() if pd.notnull(r["locality"]) else r["locality"]
        # populate state and city fields
        df.at[i, "city"] = r["locality"].split(",")[0] if pd.notnull(r["locality"]) else r["city"]
        df.at[i, "state"] = r["locality"].split(",")[1] if pd.notnull(r["locality"]) else r["state"]
        # create language field
        lang = nlp_utils.detect_lang(nlp, str(r["linkedin url"]))
        df.at[i, "lang"] = lang["language"]
    df.to_pickle(os.path.join(outpath, "feat_eng_df.pkl"))
    print(f"{datetime.now()} - Finished!")
    return df


if __name__ == "__main__":
    df = load_data()
    c_df = clean_dataset(df)
    e_df = feature_engineering(c_df)
    print(e_df.columns)
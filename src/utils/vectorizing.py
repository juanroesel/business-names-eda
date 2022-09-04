import spacy
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
nlp = spacy.load("en_core_web_trf")

artifacts_path = os.path.abspath("artifacts")

def load_batch(path):
    return pd.read_pickle(os.path.join(artifacts_path, path + ".pkl"))

def encode_vector(text, nlp):
    with nlp.select_pipes(disable=['ner', 'lemmatizer']):
        doc = nlp(text)
        return doc._.trf_data.tensors[-1]

def vectorize(data_df, vector_df, nlp, outfile):
    for i, r in tqdm(data_df.iterrows()):
        df = pd.DataFrame(encode_vector(r["linkedin url"], nlp))
        vector_df = pd.concat([vector_df, df])
    vector_df = vector_df.set_index(data_df.index[:len(data_df)])
    vector_df.to_pickle(os.path.join(artifacts_path, outfile))
    return vector_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_no", type=int)
    args = parser.parse_args()

    vector_df = pd.DataFrame()
    nlp = spacy.load("en_core_web_trf")
    print(f"{datetime.now()} - Encoding BERT vectors...")
    batch_df = load_batch(f"batch_{args.batch_no}")
    vectorize(batch_df, vector_df, nlp, f"vectors_b{args.batch_no}.pkl")
    print(f"{datetime.now()} - Finished!")

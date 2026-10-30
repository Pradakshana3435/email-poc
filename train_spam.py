import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.preprocess import preprocess

def split3(df):
    a, rest = train_test_split(df, test_size=0.30, random_state=42)
    b, c = train_test_split(rest, test_size=0.50, random_state=42)
    return a, b, c

def rd(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)

ham = [rd(f"data/{n}.csv")[["subject", "body"]] for n in ("train", "val", "test")]
hard = split3(rd("data/spam/hard_ham.csv")[["subject", "body"]])
spam = split3(rd("data/spam/spam.csv")[["subject", "body"]])

def build(i):
    h = pd.concat([ham[i], hard[i]]); h["y"] = 0
    s = spam[i].copy(); s["y"] = 1
    d = pd.concat([h, s], ignore_index=True)
    d["text"] = [preprocess(a, b, mode="tfidf") for a, b in zip(d.subject, d.body)]
    return d

tr, va, te = build(0), build(1), build(2)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
    ("lr", LogisticRegression(C=30, max_iter=2000, class_weight="balanced")),
]).fit(tr.text, tr.y)

p = pipe.predict_proba(va.text)[:, 1]
y = va.y.values
print("threshold | spam caught | real emails wrongly blocked")
for th in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    pred = p >= th
    print(f"{th:.2f}      | {(pred & (y == 1)).sum()}/{(y == 1).sum()}"
          f"        | {(pred & (y == 0)).sum()}/{(y == 0).sum()}")

joblib.dump(pipe, "models/spam_tfidf_lr.joblib")

if "--final" in sys.argv:   # run ONCE, after thresholds are frozen
    pt = pipe.predict_proba(te.text)[:, 1]
    for th in (0.5, 0.9):
        pred, yt = pt >= th, te.y.values
        print(th, "caught", (pred & (yt == 1)).sum(), "/", (yt == 1).sum(),
              "| false blocks", (pred & (yt == 0)).sum(), "/", (yt == 0).sum())
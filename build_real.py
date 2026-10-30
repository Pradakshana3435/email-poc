import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from src.preprocess import preprocess

OUT = Path("data/spam/real")
OUT.mkdir(parents=True, exist_ok=True)


def clean(df):
    d = pd.DataFrame({"subject": df["subject"].fillna("").astype(str).str.strip(),
                      "body": df["message"].fillna("").astype(str),
                      "label": df["label"]})
    d["body"] = (d.body.str.replace(r"\S+@\S+", "name@example.com", regex=True)
                       .str.replace(r"\s+", " ", regex=True).str.strip())
    d = d[d.subject != ""]
    n = d.body.str.split().str.len()
    d = d[(n >= 15) & (n <= 150)]          # similar length to your own emails
    key = (d.subject + " " + d.body).str.lower()
    return d[~key.duplicated()].reset_index(drop=True)


def take(d, label, n):
    part = d[d.label == label]
    return part.sample(min(n, len(part)), random_state=42)[["subject", "body"]]


ds = load_dataset("SetFit/enron_spam")
train, test = clean(ds["train"].to_pandas()), clean(ds["test"].to_pandas())

files = {
    "real_train_spam.csv": take(train, 1, 150),
    "real_train_ham.csv": take(train, 0, 150),
    "real_test_spam.csv": take(test, 1, 50),
    "real_test_ham.csv": take(test, 0, 50),
}
for name, d in files.items():
    d.to_csv(OUT / name, index=False, encoding="utf-8")
    print(f"{name}: {len(d)} rows, median {d.body.str.split().str.len().median():.0f} words")

# ---- evaluate the CURRENT (v1) spam model on the real test files. Nothing is trained here.
pipe = joblib.load("models/spam_tfidf_lr.joblib")


def scores(d):
    return np.array([pipe.predict_proba([preprocess(s, b, mode="tfidf")])[0][1]
                     for s, b in zip(d.subject, d.body)])


sp, hm = scores(files["real_test_spam.csv"]), scores(files["real_test_ham.csv"])
print("\nv1 model on real Enron test emails (never used in training)")
print("threshold | real spam caught | real normal emails flagged")
for th in (0.5, 0.7, 0.8, 0.85, 0.9):
    print(f"{th:.2f}      | {(sp >= th).sum()}/{len(sp)}            | {(hm >= th).sum()}/{len(hm)}")
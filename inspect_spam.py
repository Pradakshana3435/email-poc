import joblib
import numpy as np
import pandas as pd
from src.preprocess import preprocess

pipe = joblib.load("models/spam_tfidf_lr.joblib")
names = pipe.steps[0][1].get_feature_names_out()
coef = pipe.steps[-1][1].coef_[0]
print("SPAM words:", list(names[np.argsort(coef)[::-1][:30]]))
print("HAM words :", list(names[np.argsort(coef)[:30]]))

def score(s, b):
    return float(pipe.predict_proba([preprocess(s, b, mode="tfidf")])[0][1])

val = pd.read_csv("data/val.csv", dtype=str, keep_default_na=False)
before = np.array([score(s, b) for s, b in zip(val.subject, val.body)])
after = np.array([score(s, b + " Here is the page: http://example.com/order/4471")
                  for s, b in zip(val.subject, val.body)])
print(f"\nNormal val emails: {len(val)}")
print(f"score >= 0.50 without link: {(before >= 0.5).sum()}   with a link added: {(after >= 0.5).sum()}")
print(f"score >= 0.80 without link: {(before >= 0.8).sum()}   with a link added: {(after >= 0.8).sum()}")
print(f"average score without link: {before.mean():.3f}   with a link added: {after.mean():.3f}")
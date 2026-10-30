"""
train_spam_v2.py - spam model v2: synthetic + real (Enron) training data, link-neutralised.

  python train_spam_v2.py            # select C on validation only; test sets untouched
  python train_spam_v2.py --final    # ONE-TIME: score synthetic test + real_test_* for v1 and v2

Rules:
  * C is chosen on validation only (synthetic val + a real val carved from real_train_*).
  * real_test_* and the synthetic test split are only read with --final. Don't tune after that.
  * Saves to models/spam_v2_tfidf_lr.joblib, so v1 stays intact for the comparison.
ASSUMPTION: text is built with preprocess(..., mode="tfidf"). If src/spam.py prepares text
differently, change text_of() to match, or v2 won't be a drop-in for spam_check().
"""
import argparse
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.preprocess import preprocess

SEED = 42
THRESHOLDS = [0.50, 0.70, 0.80, 0.85, 0.90]
URL_RE = re.compile(r"https?://\S+|www\.\S+")
BENIGN_LINKS = ["https://www.example.com/orders/status", "https://support.example.org/tickets/open",
                "https://portal.example.net/account", "https://www.example.com/help/returns",
                "https://track.example.com/shipment"]
D = ROOT / "data"


def load(path, y, src):
    df = pd.read_csv(path).fillna("")
    if "body" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "body"})
    if "subject" not in df.columns:
        df["subject"] = ""
    return pd.DataFrame({"subject": df["subject"].astype(str), "body": df["body"].astype(str),
                         "y": y, "src": src})


def split3(df, seed=SEED):
    tr, rest = train_test_split(df, test_size=0.30, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, random_state=seed)
    return tr, va, te


def split2(df, seed=SEED):
    return train_test_split(df, test_size=0.20, random_state=seed)


def augment_links(df, p=0.4, seed=SEED):
    """Break the 'has a link = spam' shortcut: add benign links to ham, strip links from spam."""
    rng = random.Random(seed)
    out = df.copy()
    for i, r in out.iterrows():
        if rng.random() >= p:
            continue
        if r["y"] == 0:
            out.at[i, "body"] = r["body"] + " " + rng.choice(BENIGN_LINKS)
        else:
            out.at[i, "body"] = URL_RE.sub("", r["body"])
    return pd.concat([df, out.loc[out["body"] != df["body"]]], ignore_index=True)


def text_of(df):
    return [preprocess(s, b, mode="tfidf") for s, b in zip(df["subject"], df["body"])]


def fit(train, C):
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    model.fit(text_of(train), train["y"])
    return model


def report(name, scores, y):
    y = np.asarray(y)
    scores = np.asarray(scores)
    ns, nh = int((y == 1).sum()), int((y == 0).sum())
    print(f"\n{name}  ({ns} spam / {nh} ham)")
    print(" thr   spam caught   ham flagged")
    for t in THRESHOLDS:
        print(f" {t:.2f}  {int(((scores >= t) & (y == 1)).sum()):>3}/{ns:<3}       "
              f"{int(((scores >= t) & (y == 0)).sum()):>3}/{nh}")


def main(final):
    # ---- synthetic
    sp_tr, sp_va, sp_te = split3(load(D / "spam" / "spam.csv", 1, "syn_spam"))
    hh_tr, hh_va, hh_te = split3(load(D / "spam" / "hard_ham.csv", 0, "hard_ham"))
    ham_tr = load(D / "train.csv", 0, "ham")
    ham_va = load(D / "val.csv", 0, "ham")
    ham_te = load(D / "test.csv", 0, "ham")
    # ---- real (train only; real_test_* stays held out)
    rs_tr, rs_va = split2(load(D / "spam" / "real" / "real_train_spam.csv", 1, "real_spam"))
    rh_tr, rh_va = split2(load(D / "spam" / "real" / "real_train_ham.csv", 0, "real_ham"))

    train = pd.concat([sp_tr, hh_tr, ham_tr, rs_tr, rh_tr], ignore_index=True)
    val = pd.concat([sp_va, hh_va, ham_va, rs_va, rh_va], ignore_index=True)
    train = augment_links(train)
    print(f"train {len(train)} (spam {int(train.y.sum())}), val {len(val)} (spam {int(val.y.sum())})")

    # ---- choose C on validation log loss
    best = None
    for C in [1, 10, 30, 100, 300, 1000]:
        m = fit(train, C)
        ll = log_loss(val["y"], m.predict_proba(text_of(val))[:, 1])
        print(f"C={C:<4} val log loss {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, C)
    C = best[1]
    print(f"chosen C = {C}")

    m = fit(train, C)
    vs = m.predict_proba(text_of(val))[:, 1]
    report("VALIDATION (all)", vs, val["y"])
    real_va = val["src"].isin(["real_spam", "real_ham"])
    report("VALIDATION (real only)", vs[real_va], val.loc[real_va, "y"])

    # ---- link-shortcut check: same ham with and without a benign link
    h = val[val.y == 0].head(40)
    plain = m.predict_proba(text_of(h))[:, 1].mean()
    linked = h.assign(body=h["body"] + " " + BENIGN_LINKS[0])
    print(f"\nlink check: mean spam score on ham  plain {plain:.3f}  +link "
          f"{m.predict_proba(text_of(linked))[:, 1].mean():.3f}  (should be close)")
    bad = val.assign(score=vs, snip=val["body"].str[:80])
    print("\nham scoring >= 0.50:")
    print(bad[(bad.y == 0) & (bad.score >= 0.5)][["src", "score", "subject", "snip"]].to_string())
    print("\nspam scoring < 0.50 (first 15):")
    print(bad[(bad.y == 1) & (bad.score < 0.5)][["src", "score", "subject", "snip"]].head(15).to_string())
    if not final:
        print("\nTest sets untouched. Run with --final once you are happy with the choices above.")
        return

    # ---- refit on train + val, then ONE look at the test sets
    full = augment_links(pd.concat([sp_tr, hh_tr, ham_tr, rs_tr, rh_tr, sp_va, hh_va, ham_va, rs_va, rh_va],
                                   ignore_index=True))
    m2 = fit(full, C)
    joblib.dump(m2, ROOT / "models" / "spam_v2_tfidf_lr.joblib")

    syn_te = pd.concat([sp_te, hh_te, ham_te], ignore_index=True)
    report("v2  SYNTHETIC TEST", m2.predict_proba(text_of(syn_te))[:, 1], syn_te["y"])

    real_te = pd.concat([load(D / "spam" / "real" / "real_test_spam.csv", 1, "real_spam"),
                         load(D / "spam" / "real" / "real_test_ham.csv", 0, "real_ham")], ignore_index=True)
    report("v2  REAL TEST (Enron)", m2.predict_proba(text_of(real_te))[:, 1], real_te["y"])

    try:  # v1 on the same real test, for the comparison
        from src.spam import spam_check
        v1 = [spam_check(s, b)[0] for s, b in zip(real_te["subject"], real_te["body"])]
        report("v1  REAL TEST (Enron)", v1, real_te["y"])
    except Exception as e:
        print("v1 comparison skipped:", e)
    print("\nDone. Do not tune on these sets again. A data change needs a new split and a new --final.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    main(ap.parse_args().final)

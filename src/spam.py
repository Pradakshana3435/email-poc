from pathlib import Path
import joblib
import numpy as np
from src.preprocess import preprocess
from src.decision import load_config

ROOT = Path(__file__).resolve().parents[1]
_P = {}


def _pipe():
    if "pipe" not in _P:
        cfg = load_config(ROOT / "config" / "pipeline.yaml").get("spam", {})
        rel = cfg.get("model_path", "models/spam_tfidf_lr.joblib")
        _P["pipe"] = joblib.load(ROOT / rel)
    return _P["pipe"]


def spam_check(subject, body, k=5):
    pipe = _pipe()
    t = preprocess(subject, body, mode="tfidf")
    score = float(pipe.predict_proba([t])[0][1])
    vec, clf = pipe.steps[0][1], pipe.steps[-1][1]
    contrib = vec.transform([t]).toarray()[0] * clf.coef_[0]
    names = vec.get_feature_names_out()
    terms = [(names[i], round(float(contrib[i]), 2))
             for i in np.argsort(contrib)[::-1][:k] if contrib[i] > 0]
    return score, terms
#spam.py
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
        pipe = joblib.load(ROOT / rel)
        _P["vec"], _P["clf"] = pipe.steps[0][1], pipe.steps[-1][1]
        _P["names"] = _P["vec"].get_feature_names_out()   # built once, not on every email
        _P["coef"] = _P["clf"].coef_[0]
        _P["pipe"] = pipe
    return _P["pipe"]


def spam_check(subject, body, k=5):
    _pipe()
    t = preprocess(subject, body, mode="tfidf")
    x = _P["vec"].transform([t])                          # sparse row, transformed once
    score = float(_P["clf"].predict_proba(x)[0][1])
    idx = x.indices                                       # only the words that are in this email
    contrib = x.data * _P["coef"][idx]
    order = np.argsort(contrib)[::-1][:k]
    terms = [(_P["names"][idx[i]], round(float(contrib[i]), 2)) for i in order if contrib[i] > 0]
    return score, terms
#predict.py
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from src.notices import detect_notice

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess
from src.decision import load_config, decide
from src.priority import hybrid_priority

MODELS = ROOT / "models"
# display-only filter: hide date artifacts from the explanation
HIDE = {"num", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "january", "february", "march", "april", "june", "july", "august",
        "september", "october", "november", "december"}
_S = {}


def _load():
    if _S:
        return _S
    from sentence_transformers import SentenceTransformer
    meta = joblib.load(MODELS / "meta.joblib")
    _S.update(
        cfg=load_config(ROOT / "config" / "pipeline.yaml"),
        lr_t=joblib.load(MODELS / "tfidf_lr.joblib"),
        lr_e=joblib.load(MODELS / "minilm_lr.joblib"),
        pri=joblib.load(MODELS / "priority_model.joblib"),
        T=meta["T"],
        classes=np.array(meta["classes"]),
        E_tr=normalize(np.load(MODELS / "E_train.npy")),
        train=pd.read_csv(ROOT / "data" / "train.csv"),
        st=SentenceTransformer("all-MiniLM-L6-v2"),
    )
    return _S


def _sharpen(P, T):
    Q = np.power(P, 1.0 / T)
    return Q / Q.sum(axis=1, keepdims=True)


def _top_terms(s, text_t, cls, k=5):
    vec, clf = s["lr_t"].steps[0][1], s["lr_t"].steps[-1][1]
    x = vec.transform([text_t]).toarray()[0]
    contrib = x * clf.coef_[list(clf.classes_).index(cls)]
    names = vec.get_feature_names_out()
    out = []
    for i in np.argsort(contrib)[::-1]:
        if contrib[i] <= 0:
            break
        if set(names[i].split()) & HIDE:
            continue
        out.append((names[i], round(float(contrib[i]), 2)))
        if len(out) == k:
            break
    return out


def predict(subject, body):
    subject, body = subject or "", body or ""

    is_notice, markers = detect_notice(subject, body)
    if is_notice:
        return dict(category="Unknown", predicted=None, confidence=0.0, margin=0.0, similarity=None,
                    department="Filtered (automated)", action="filtered",
                    reason="automated notice: " + ", ".join(markers),
                    priority="NA", priority_reason="not applicable (automated notice)",
                    top_terms=[], neighbours=[])

    s = _load()
    t = preprocess(subject, body, mode="tfidf")
    e = preprocess(subject, body, mode="embedding")
    emb = s["st"].encode([e])

    P = _sharpen((s["lr_t"].predict_proba([t]) + s["lr_e"].predict_proba(emb)) / 2, s["T"])[0]
    sims = s["E_tr"] @ normalize(emb)[0]
    d = decide(P, s["classes"], s["cfg"], sim=float(sims.max()))

    ml_pri = s["pri"].predict([t])
    pri, why = hybrid_priority(pd.DataFrame({"subject": [subject], "body": [body]}), ml_pri)

    top = np.argsort(sims)[::-1][:3]
    is_unknown = d["category"] == "Unknown"
    d.update(
        priority="NA" if is_unknown else str(pri[0]),
        priority_reason="not applicable (Unknown)" if is_unknown else (why[0] or "model"),    
        top_terms=_top_terms(s, t, d["predicted"]),
        neighbours=[(s["train"].category.iloc[j], round(float(sims[j]), 2), s["train"].subject.iloc[j])
                    for j in top],
    )
    return d
#decision.py
import yaml
import numpy as np
from pathlib import Path


def load_config(path="config/pipeline.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def decide(proba, classes, cfg, sim=None):
    """proba: 1-D class probabilities aligned with `classes`.
    sim: max cosine similarity of the email to the training emails (optional)."""
    order = np.argsort(proba)[::-1]
    top1, top2 = proba[order[0]], proba[order[1]]
    pred, margin = classes[order[0]], top1 - top2
    t = cfg["thresholds"]

    reason = None
    if top1 < t["review_flag"]:
        reason = "low confidence"
    elif sim is not None and sim < t.get("min_similarity", 0.0):
        reason = "unlike training emails"
    elif margin < t.get("min_margin", 0.0):
        reason = "two categories too close"

    if reason:
        return dict(category="Unknown", predicted=pred, confidence=top1, margin=margin,
                    similarity=sim, department=cfg["routing"]["Unknown"],
                    action="manual_review", reason=reason)

    flagged = top1 < t["auto_route"]
    return dict(category=pred, predicted=pred, confidence=top1, margin=margin,
                similarity=sim, department=cfg["routing"][pred],
                action="route_with_flag" if flagged else "auto_route",
                reason="medium confidence" if flagged else None)
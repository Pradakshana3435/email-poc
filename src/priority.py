import numpy as np
import pandas as pd

RULES = {
 "urgent":     r"\burgent(?:ly)?\b|\bemergency\b",
 "asap":       r"\basap\b|as soon as possible|\bimmediate(?:ly)?\b|right away|right now|at once",
 "escalation": r"escalat|legal action|consumer (?:forum|court)|regulator|social media|public review|lawyer|chargeback",
 "security":   r"\bhack|unauthori[sz]ed|\bsomeone (?:changed|logged|accessed|else)|isn't mine|\bnot me\b|suspicious|compromis|\bscam|fraud|stolen",
 "money_lost": r"(?:charged|debited|billed)(?: me)? (?:twice|two times|2x)|double (?:charged|debited|billing)|(?:deducted|debited|taken out|gone from)|money (?:was )?(?:deducted|debited|gone|taken)",
}
NEG = r"\b(?:not|no|isn't|nothing)\b[^.]{0,15}\b(?:urgent|rush|hurry)\b|low priority|whenever (?:you|it)"
KEEP = ["asap", "escalation", "money_lost", "urgent", "security"]


def rule_hits(df):
    t = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.lower()
    H = pd.DataFrame({k: t.str.contains(v, regex=True) for k, v in RULES.items()})
    H.loc[t.str.contains(NEG, regex=True), ["urgent", "asap"]] = False
    return H


def hybrid_priority(df, ml_pred):
    """Rules can only upgrade to High; otherwise the ML prediction stands."""
    H = rule_hits(df)[KEEP]
    fired = H.any(axis=1).values
    why = H.apply(lambda r: ", ".join(r.index[r]), axis=1).values
    return np.where(fired, "High", ml_pred), why
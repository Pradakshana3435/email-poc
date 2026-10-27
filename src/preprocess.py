"""Reusable text preprocessing. Import this in BOTH training and prediction code.

    from preprocess import preprocess, preprocess_df
    preprocess("Re: Payment failed", "<p>My payment was deducted...</p> Thanks, Sara")
"""
import re
import html
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# negations carry meaning ("payment NOT received"), so they are never removed
KEEP = {"no", "not", "nor", "never", "none", "cannot", "nothing", "without", "against"}
STOP = set(ENGLISH_STOP_WORDS) - KEEP

_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PREFIX = re.compile(r"^\s*((re|fw|fwd)\s*:\s*)+", re.I)
_SIGN = re.compile(
    r"\b(?:thanks|thank you|many thanks|regards|best regards|kind regards|warm regards|"
    r"sincerely|yours sincerely|yours faithfully|cheers|best wishes)\b"
    r"[\s,.!-]*(?:[\w.:'#-]+\s*){0,6}$", re.I)
_NUM = re.compile(r"\d[\d,.]*")


def light_clean(text):
    """HTML, URLs, emails, signature and whitespace only. Keeps case and stop words
    (the right input for a sentence-transformer model)."""
    t = html.unescape(text if isinstance(text, str) else "")
    t = _HTML.sub(" ", t)
    t = _URL.sub(" url ", t)
    t = _EMAIL.sub(" emailaddr ", t)
    t = _SIGN.sub(" ", t.strip())
    return re.sub(r"\s+", " ", t).strip()


def clean_text(text, remove_stop=True):
    """Full cleaning for TF-IDF: lowercase, numbers -> 'num', punctuation removed, tokenised."""
    t = light_clean(text).lower()
    t = _NUM.sub(" num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = [w for w in t.split() if len(w) > 1]
    if remove_stop:
        tokens = [w for w in tokens if w not in STOP]
    return " ".join(tokens)


def combine(subject, body):
    """Subject + body as one text (a leading 'Re:' / 'Fwd:' in the subject is dropped)."""
    s = subject if isinstance(subject, str) else ""
    b = body if isinstance(body, str) else ""
    return _PREFIX.sub("", s) + ". " + b


def preprocess(subject, body, mode="tfidf"):
    """mode='tfidf' -> cleaned token string; mode='embedding' -> lightly cleaned text."""
    text = combine(subject, body)
    return light_clean(text) if mode == "embedding" else clean_text(text)


def preprocess_df(df, mode="tfidf"):
    return [preprocess(s, b, mode) for s, b in zip(df["subject"], df["body"])]

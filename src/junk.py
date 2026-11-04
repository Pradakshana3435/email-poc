"""Cheap checks that stop obvious junk before it reaches any model.

The category model always picks one of its eight categories, so gibberish has to be caught here.
The rules are deliberately conservative: a real email rejected is worse than a junk one let through.
They do NOT understand meaning. A well-formed but irrelevant email (phishing, lottery, marketing)
passes this filter and is the job of the spam model.

Works on letters in any script (English, Tamil, Tanglish, ...).
"""
import re
import unicodedata

_MASH = ("qwert", "asdf", "zxcv", "hjkl", "sdfg", "dfgh", "xcvb")
_FILLER = {"hi", "hello", "hey", "dear", "sir", "madam", "team", "test", "testing", "sample", "ok", "okay",
           "thanks", "thank", "you", "please", "pls", "help", "regards", "good", "morning", "afternoon", "evening"}
_RUN = re.compile(r"([A-Za-z0-9])\1{5,}")  # same letter or digit six or more times ("aaaaaaa")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")

def _mostly_runs(text):
    """True only when repeated-character runs make up a big share of the text.
    Tokens containing digits (account numbers like 98XXXXXX61, order IDs) are ignored."""
    kept = " ".join(t for t in text.split() if not any(c.isdigit() for c in t))
    visible = sum(1 for c in kept if not c.isspace())
    if visible == 0:
        return False
    run_chars = sum(len(m.group(0)) for m in _RUN.finditer(kept))
    return run_chars > 0.3 * visible

def _letters(token):
    """Letters AND combining marks (Tamil vowel signs are marks, not letters)."""
    return [c for c in token if unicodedata.category(c)[0] in ("L", "M")]


def normalise(text):
    """Lower-case, single-spaced text. Use it to spot the same email sent twice."""
    return " ".join((text or "").lower().split())


def junk_reason(subject, body):
    """None if the email can pass, otherwise a short reason string."""
    text = f"{subject or ''} {body or ''}"
    text = _EMAIL.sub(" ", _URL.sub(" ", text))  # links and addresses say nothing about the wording
    tokens = text.split()
    letters = [_letters(t) for t in tokens]
    words = [l for l in letters if len(l) >= 2]
    total_letters = sum(len(l) for l in letters)
    visible = sum(1 for c in text if not c.isspace())

    if total_letters < 6 or len(words) < 2:
        return "too short to classify"
    if total_letters / max(1, visible) < 0.5:
        return "mostly digits and symbols"
    if _mostly_runs(text):
        return "mostly repeated characters"
    lowered = ["".join(w).lower() for w in words]
    if all(w in _FILLER for w in lowered):  # "hi hello", "test", "please help": nothing to classify
        return "greeting only, no content"
    if len(lowered) >= 6 and len(set(lowered)) / len(lowered) < 0.34:
        return "the same words repeated"
    latin = [w for w in lowered if len(w) >= 4 and w.isascii() and w.isalpha()]
    if len(latin) >= 2:  # keyboard mashing: no vowels, or keys that sit side by side
        mash = [w for w in latin if not any(v in w for v in "aeiouy") or any(m in w for m in _MASH)]
        if len(mash) / len(latin) > 0.5:
            return "looks like keyboard mashing"
    return None


def junk_result(reason):
    """The dict predict() returns for junk, in the same shape as the other early exits."""
    return dict(category="Unknown", predicted=None, confidence=0.0, margin=0.0, similarity=None,
                department="Filtered (junk)", action="junk",
                reason="junk: " + reason,
                priority="NA", priority_reason="not applicable (junk)",
                top_terms=[], neighbours=[])
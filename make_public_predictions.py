#make_public_predictions.py
"""Make shareable copies of the final prediction files, with the email text removed.

The hand-written emails include some from a personal inbox, so the full prediction
files (which contain subject and body) must not be published. The *_public.csv
versions keep everything needed to check the evaluation (id, true label, prediction,
action, confidence, priority) and drop only the text.

Run from the email-poc folder:  python make_public_predictions.py
"""
from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parent / "reports"
TEXT_COLUMNS = ["subject", "body"]

for name in ["final_test_predictions", "final_handwritten_predictions"]:
    src = REPORTS / f"{name}.csv"
    if not src.exists():
        print(f"skipped {src.name} (not found)")
        continue
    df = pd.read_csv(src, keep_default_na=False)
    public = df.drop(columns=[c for c in TEXT_COLUMNS if c in df.columns])
    dst = REPORTS / f"{name}_public.csv"
    public.to_csv(dst, index=False)
    print(f"wrote {dst.name}: {len(public)} rows, columns = {list(public.columns)}")

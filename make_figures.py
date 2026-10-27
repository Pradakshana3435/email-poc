#make_figure.py
"""Regenerate the confusion matrices for the report from the saved prediction files.

Reads the saved prediction files in reports/ (the full version if it exists, otherwise
the *_public.csv version with the text removed). Does NOT run any model, so the
frozen evaluation is untouched.

Run from the email-poc folder:  python make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

LABELS = [
    "Technical Support", "Payment Issue", "Account & Login", "Sales Inquiry",
    "Complaint", "Refund Request", "Job Application", "General Inquiry", "Unknown",
]


def plot(stem, out_name, title):
    path = REPORTS / f"{stem}.csv"
    if not path.exists():
        path = REPORTS / f"{stem}_public.csv"
    df = pd.read_csv(path, keep_default_na=False)
    cm = confusion_matrix(df["category"], df["pred"], labels=LABELS)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
        ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted (Unknown = sent to manual review)")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(REPORTS / out_name, dpi=150)
    plt.close(fig)
    print(f"saved reports/{out_name}  ({len(df)} emails)")


plot("final_test_predictions", "cm_final_test.png",
     "Held-out test (152 emails)")
plot("final_handwritten_predictions", "cm_final_handwritten.png",
     "Hand-written test (82 emails)")

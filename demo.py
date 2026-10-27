"""Terminal demo of the email classification and routing system.

Run from the email-poc folder (virtual environment active):
    python demo.py

It does three things:
  1. classifies a few example emails one at a time,
  2. classifies a whole CSV (data/batch_demo.csv), and
  3. shows the review queue: the emails that need a person.

Results go to data/demo_results.db, a throwaway database that is recreated on every
run, so the demo never touches the dashboard's data/results.db.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.predict import predict  # noqa: E402  (loads the models, takes a few seconds)

DB = ROOT / "data" / "demo_results.db"
BATCH = ROOT / "data" / "batch_demo.csv"

EXAMPLES = [
    ("Payment failed",
     "Hi, my card was charged yesterday but the order page says the payment failed. "
     "Can you check what happened to my money?"),
    ("Application for data analyst role",
     "Dear Hiring Manager, I am writing to apply for the data analyst position. "
     "My CV is attached. I have two years of experience with Python and SQL."),
    ("URGENT: dashboard is down",
     "Nobody on our team can open the dashboard since this morning and a client demo "
     "is at 3 PM. Please fix this ASAP."),
    ("Do you offer student discounts?",
     "Hello, I wanted to know whether you have any pricing for students or small teams."),
    ("Weekly newsletter",
     "This is an automated message. You are receiving this email because you subscribed "
     "to our newsletter. Click here to unsubscribe."),
]


def line(r):
    conf = "n/a" if r["category"] == "Unknown" else f"{float(r['confidence']):.0%}"
    return (f"{r['category']:<18} | {r['department']:<16} | {r['action']:<15} | "
            f"conf {conf:>4} | priority {r['priority']}")


def connect():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT DEFAULT CURRENT_TIMESTAMP,
        subject TEXT, body TEXT, category TEXT, department TEXT,
        action TEXT, confidence REAL, priority TEXT, reason TEXT, reviewed INTEGER DEFAULT 0)""")
    return con


def save(con, subject, body, r):
    con.execute(
        "INSERT INTO results (subject, body, category, department, action, confidence, "
        "priority, reason) VALUES (?,?,?,?,?,?,?,?)",
        (subject, body, r.get("category"), r.get("department"), r.get("action"),
         float(r.get("confidence") or 0), str(r.get("priority")), str(r.get("reason") or "")))


def heading(text):
    print("\n" + text)
    print("-" * len(text))


def main():
    if DB.exists():
        DB.unlink()
    con = connect()

    heading("1. Single emails")
    for subject, body in EXAMPLES:
        r = predict(subject, body)
        save(con, subject, body, r)
        print(f"\n\"{subject}\"")
        print("  " + line(r))
        if r.get("priority_reason"):
            print(f"  priority from: {r['priority_reason']}")
        if r.get("reason"):
            print(f"  note: {r['reason']}")
        terms = ", ".join(t for t, _ in (r.get("top_terms") or []))
        if terms:
            print(f"  words that pushed the prediction: {terms}")
        for cat, sim, subj in (r.get("neighbours") or [])[:2]:
            print(f"  similar training email: {sim:.2f} | {cat} | {subj}")

    heading("2. Batch (data/batch_demo.csv)")
    if not BATCH.exists():
        print("data/batch_demo.csv not found, skipping.")
    else:
        df = pd.read_csv(BATCH, dtype=str, keep_default_na=False)
        rows = []
        for s, b in zip(df["subject"], df["body"]):
            r = predict(s, b)
            save(con, s, b, r)
            rows.append({"subject": s[:45], "category": r["category"],
                         "action": r["action"], "priority": r["priority"]})
        out = pd.DataFrame(rows)
        print(f"{len(out)} emails classified.\n")
        print(out.to_string(index=False))
        print("\nBy action:", out["action"].value_counts().to_dict())

    heading("3. Review queue (manual review or routed with a flag)")
    con.commit()
    q = pd.read_sql(
        "SELECT id, subject, category, action, confidence FROM results "
        "WHERE action IN ('manual_review', 'route_with_flag') AND reviewed = 0 "
        "ORDER BY id", con)
    con.close()
    if q.empty:
        print("Nothing needs review in this run.")
    else:
        q["subject"] = q["subject"].str.slice(0, 45)
        q["confidence"] = q["confidence"].round(2)
        print(f"{len(q)} emails waiting for a person:\n")
        print(q.to_string(index=False))
    print("\nFor the full dashboard: streamlit run app/app.py")


if __name__ == "__main__":
    main()

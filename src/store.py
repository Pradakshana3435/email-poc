#store.py
import json
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "results.db"


def save(subject, body, r):
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT DEFAULT CURRENT_TIMESTAMP,
        subject TEXT, body TEXT, category TEXT, model_guess TEXT,
        confidence REAL, priority TEXT, department TEXT,
        action TEXT, reason TEXT, explanation TEXT)""")
    con.execute(
        "INSERT INTO results (subject, body, category, model_guess, confidence, priority, "
        "department, action, reason, explanation) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (subject, body, r["category"], r["predicted"], float(r["confidence"]), r["priority"],
         r["department"], r["action"], str(r.get("reason") or ""),
         json.dumps({"terms": r["top_terms"], "neighbours": r["neighbours"],
                     "priority_reason": r["priority_reason"]})))
    con.commit()
    con.close()
#app.py
import io
import sys
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import predict
from src.decision import load_config

DB = ROOT / "data" / "results.db"
CATS = load_config(ROOT / "config" / "pipeline.yaml")["categories"] + ["Unknown"]


def connect():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT DEFAULT CURRENT_TIMESTAMP,
        subject TEXT, body TEXT, category TEXT, model_guess TEXT,
        confidence REAL, priority TEXT, department TEXT,
        action TEXT, reason TEXT, explanation TEXT)""")
    have = {row[1] for row in con.execute("PRAGMA table_info(results)")}
    for name, ddl in [("reviewed", "INTEGER DEFAULT 0"), ("final_category", "TEXT")]:
        if name not in have:
            con.execute(f"ALTER TABLE results ADD COLUMN {name} {ddl}")
    return con


def save(subject, body, r):
    con = connect()
    con.execute(
        "INSERT INTO results (subject, body, category, model_guess, confidence, priority, "
        "department, action, reason, explanation) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (subject, body, r.get("category"), r.get("predicted"),
         float(r.get("confidence") or 0), str(r.get("priority")), r.get("department"),
         r.get("action"), str(r.get("reason") or ""),
         json.dumps({"terms": r.get("top_terms"), "neighbours": r.get("neighbours"),
                     "priority_reason": r.get("priority_reason")}, default=str)))
    con.commit()
    con.close()


def show_result(r):
    if r["action"] == "filtered":
        st.info(f"Filtered as an automated notice, no action needed ({r.get('reason')}). "
                "It is saved and can be reviewed later.")
        return

    if r["action"] == "auto_route":
        st.success(f"Auto-routed to {r['department']}")
    elif r["action"] == "route_with_flag":
        st.warning(f"Routed to {r['department']} with a review flag ({r.get('reason')})")
    else:
        st.error(f"Manual review needed ({r.get('reason')})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category", r["category"])
    c2.metric("Department", r["department"])
    c3.metric("Priority", r["priority"])
    c4.metric("Confidence", "n/a" if r["category"] == "Unknown" else f"{r['confidence']:.0%}")
    if r["category"] == "Unknown":
        st.caption(f"Closest category was {r['predicted']} ({r['confidence']:.0%} confident), "
                   "but the email looks unlike the training emails.")
    st.caption(f"Priority from: {r.get('priority_reason')}")

    st.subheader("Why this result")
    left, right = st.columns(2)
    with left:
        st.markdown("**Words that pushed the prediction**")
        st.write(", ".join(t for t, _ in (r.get("top_terms") or [])) or "none found")
    with right:
        st.markdown("**Most similar training emails**")
        for cat, sim, subj in (r.get("neighbours") or []):
            st.write(f"{sim:.2f} | {cat} | {subj}")


st.set_page_config(page_title="Email triage", layout="wide")
st.title("Email classification and routing")
tab_one, tab_batch, tab_review, tab_hist = st.tabs(
    ["Single email", "Batch (CSV)", "Review queue", "History"])

# ------------------------------------------------------------ single email
with tab_one:
    subject = st.text_input("Subject")
    body = st.text_area("Body", height=200)
    if st.button("Classify", type="primary"):
        if not body.strip():
            st.warning("Please enter an email body.")
            st.stop()
        with st.spinner("Classifying..."):
            r = predict(subject, body)
        save(subject, body, r)
        show_result(r)

# --------------------------------------------------------------- batch CSV
with tab_batch:
    st.write("Upload a CSV with `subject` and `body` columns.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up).fillna("")
        missing = {"subject", "body"} - set(df.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
        elif st.button("Classify all emails"):
            bar, rows = st.progress(0.0), []
            for i, (s, b) in enumerate(zip(df["subject"], df["body"])):
                r = predict(str(s), str(b))
                save(str(s), str(b), r)
                rows.append({"subject": s, "category": r.get("category"),
                             "department": r.get("department"), "action": r.get("action"),
                             "confidence": round(float(r.get("confidence") or 0), 3),
                             "priority": r.get("priority")})
                bar.progress((i + 1) / len(df))
            out = pd.DataFrame(rows)
            st.dataframe(out, hide_index=True)
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("Download results", buf.getvalue(), "results.csv", "text/csv")

# ------------------------------------------------------------ review queue
with tab_review:
    con = connect()
    q = pd.read_sql("SELECT id, ts, subject, body, category, model_guess, action, confidence, "
                    "reason FROM results WHERE action IN ('manual_review', 'route_with_flag') "
                    "AND COALESCE(reviewed, 0) = 0 ORDER BY id DESC", con)
    con.close()
    st.write(f"{len(q)} emails waiting for review")
    if len(q):
        st.dataframe(q.drop(columns=["body"]), hide_index=True)
        pick = st.selectbox("Open email id", q["id"])
        row = q[q["id"] == pick].iloc[0]
        st.text_area("Body", row["body"], height=140, disabled=True)
        final = st.selectbox("Correct category", CATS)
        if st.button("Mark as reviewed"):
            con = connect()
            con.execute("UPDATE results SET reviewed = 1, final_category = ? WHERE id = ?",
                        (final, int(pick)))
            con.commit()
            con.close()
            st.rerun()

# ----------------------------------------------------------------- history
with tab_hist:
    con = connect()
    h = pd.read_sql("SELECT id, ts, subject, category, department, action, confidence, "
                    "priority, reviewed, final_category FROM results ORDER BY id DESC", con)
    con.close()
    if h.empty:
        st.info("No emails classified yet.")
    else:
        a, b = st.columns(2)
        a.subheader("By action")
        a.bar_chart(h["action"].value_counts())
        b.subheader("By category")
        b.bar_chart(h["category"].value_counts())
        st.dataframe(h, hide_index=True)
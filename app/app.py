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
from src import quota
from src.predict import predict
from src.decision import load_config
from src.store import folder_of

DB = ROOT / "data" / "results.db"
CATS = load_config(ROOT / "config" / "pipeline.yaml")["categories"] + ["Unknown", "Spam"]

@st.cache_resource(show_spinner="Loading models (first start only)...")
def warm_up():
    from src import predict as P
    P._load()
    P.spam_check("warm up", "this is only a warm up message")
    return True

warm_up()

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
    if r["action"] == "junk":
        st.info(f"Not classified: {r.get('reason')}. Please paste a real email with a subject and body.")
        return
    if r["action"] == "spam":
        st.error(f"Blocked as spam ({r['confidence']:.0%} spam score). "
                 "Quarantined, not routed to any team.")
        st.markdown("**Words that triggered it**")
        st.write(", ".join(t for t, _ in (r.get("top_terms") or [])) or "none found")
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
    if r["category"] == "Unknown" and r.get("predicted"):
        st.caption(f"Closest category was {r['predicted']} ({r['confidence']:.0%} confident), "
                   "but the email looks unlike the training emails.")
    st.caption(f"Priority from: {r.get('priority_reason')}")

    st.subheader("Why this result")
    left, right = st.columns(2)
    with left:
        st.markdown("**Words that pushed the prediction**")
        st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
        st.write(", ".join(t for t, _ in (r.get("top_terms") or [])) or "none found")
    with right:
        st.markdown("**Most similar training emails**")
        for cat, sim, subj in (r.get("neighbours") or []):
            st.write(f"{sim:.2f} | {cat} | {subj}")


st.set_page_config(page_title="Email triage", layout="wide")
slot = st.sidebar.empty()
quota.render(slot)

st.title("Email classification and routing")
tab_one, tab_batch, tab_inbox, tab_review, tab_hist = st.tabs(
    ["Single email", "Batch (CSV)", "Inbox by department", "Review queue", "History"])

# ------------------------------------------------------------ single email
with tab_one:
    subject = st.text_input("Subject")
    body = st.text_area("Body", height=200)
    if st.button("Classify", type="primary"):
        if quota.remaining() == 0:
            st.error(quota.MSG)
        elif not body.strip():
            st.warning("Please enter an email body.")
        else:
            with st.spinner("Classifying..."):
                r = predict(subject, body)
            quota.consume(1)
            quota.render(slot)
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
        elif quota.remaining() == 0:
            st.error(quota.MSG)
        else:
            if len(df) > quota.remaining():
                st.warning(f"{len(df)} rows uploaded, but only {quota.remaining()} emails "
                           f"are left in this session. Only the first {quota.remaining()} "
                           "rows will be classified.")
                df = df.head(quota.remaining())
            if st.button("Classify all emails"):
                bar, rows = st.progress(0.0), []
                for i, (s, b) in enumerate(zip(df["subject"], df["body"])):
                    r = predict(str(s), str(b))
                    quota.consume(1)
                    save(str(s), str(b), r)
                    rows.append({"subject": s, "category": r.get("category"),
                                 "department": r.get("department"), "action": r.get("action"),
                                 "confidence": round(float(r.get("confidence") or 0), 3),
                                 "priority": r.get("priority")})
                    bar.progress((i + 1) / len(df))
                quota.render(slot)
                out = pd.DataFrame(rows)
                st.dataframe(out, hide_index=True)
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("Download results", buf.getvalue(), "results.csv", "text/csv")

# --------------------------------------------------------- inbox by department
with tab_inbox:
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    con = connect()
    inbox = pd.read_sql("SELECT id, ts, subject, body, category, department, action, confidence, priority, "
                        "reason, reviewed, final_category FROM results ORDER BY id DESC", con)
    con.close()
    if inbox.empty:
        st.info("No emails classified yet.")
    else:
        routing = load_config(ROOT / "config" / "pipeline.yaml")["routing"]
        inbox["reviewed"] = inbox["reviewed"].fillna(0).astype(int)
        inbox["final_category"] = inbox["final_category"].fillna("")
        inbox["folder"] = inbox.apply(lambda r: folder_of(r, routing), axis=1)
        depts = sorted(set(routing.values()) - {"Manual Review"})
        folders = depts + ["Manual Review", "Spam", "Filtered (automated)"]
        counts = inbox["folder"].value_counts().to_dict()
        chosen = st.radio("Folder", folders, horizontal=True, key="inbox_folder",
                          format_func=lambda f: f"{f} ({counts.get(f, 0)})")

        view = inbox[inbox["folder"] == chosen].copy()
        view["_p"] = view["priority"].map({"High": 0, "Medium": 1, "Low": 2}).fillna(3)
        view = view.sort_values(["_p", "id"], ascending=[True, False])
        st.write(f"{len(view)} email(s) in {chosen}")
        if len(view):
            view["status"] = view.apply(
                lambda r: f"Routed by you ({r['final_category']})" if r["reviewed"] else "", axis=1)
            show = ["id", "ts", "subject", "category", "priority", "confidence", "action", "status", "reason"]
            st.dataframe(view[show], hide_index=True)
            pick = st.selectbox("Open email id", view["id"], key="inbox_pick")
            row = view[view["id"] == pick].iloc[0]
            st.text_area("Body", row["body"], height=160, disabled=True, key="inbox_body")
            st.caption(f"Category: {row['category']} | Priority: {row['priority']} | "
                       f"Confidence: {row['confidence']:.0%} | {row['reason'] or 'no note'}")
            # ---- route this email to another department (works from any folder)
            dept_of = lambda c: "Spam" if c == "Spam" else routing.get(c, "Manual Review")
            current = row["final_category"] if row["reviewed"] and row["final_category"] else row["category"]
            if row["reviewed"]:
                st.info(f"You routed this email to {dept_of(row['final_category'])}.")            
            dest = st.selectbox("Send this email to", CATS,
                                index=CATS.index(current) if current in CATS else 0,
                                format_func=lambda c: f"{dept_of(c)}  (as {c})",
                                key=f"inbox_dest_{pick}")
            if st.button("Route email", key=f"inbox_route_{pick}"):
                con = connect()
                con.execute("UPDATE results SET reviewed = 1, final_category = ? WHERE id = ?",
                            (dest, int(pick)))
                con.commit()
                con.close()
                st.session_state["flash"] = f"Email {int(pick)} routed to {dept_of(dest)}."
                st.rerun()

# ------------------------------------------------------------ review queue
with tab_review:
    con = connect()
    q = pd.read_sql("SELECT id, ts, subject, body, category, model_guess, action, confidence, "
                    "reason FROM results WHERE action IN ('manual_review', 'route_with_flag', 'spam') "
                    "AND COALESCE(reviewed, 0) = 0 ORDER BY id DESC", con)
    con.close()
    st.write(f"{len(q)} emails waiting for review")
    if len(q):
        st.dataframe(q.drop(columns=["body"]), hide_index=True)
        pick = st.selectbox("Open email id", q["id"], key="review_pick")
        row = q[q["id"] == pick].iloc[0]
        st.text_area("Body", row["body"], height=140, disabled=True, key="review_body")
        final = st.selectbox("Correct category", CATS, key="review_final")
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
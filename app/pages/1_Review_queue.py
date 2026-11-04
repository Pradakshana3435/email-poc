#1_Review_queue.py  (read-only folder view; corrections are made in the dashboard's Review queue tab)
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.decision import load_config
from src.store import folder_of

DB = ROOT / "data" / "results.db"

st.title("Emails by folder")
st.set_page_config(page_title="Folders", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

if not DB.exists():
    st.info("No results yet. Classify an email on the main page first.")
    st.stop()

con = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM results ORDER BY id DESC", con)
con.close()
if df.empty:
    st.info("No results yet. Classify an email on the main page first.")
    st.stop()

for col, default in (("reviewed", 0), ("final_category", "")):  # older databases may lack these columns
    if col not in df.columns:
        df[col] = default
df["reviewed"] = df["reviewed"].fillna(0).astype(int)
df["final_category"] = df["final_category"].fillna("")

routing = load_config(ROOT / "config" / "pipeline.yaml")["routing"]
df["folder"] = df.apply(lambda r: folder_of(r, routing), axis=1)
folders = ["All"] + sorted(set(routing.values()) - {"Manual Review"}) + \
          ["Manual Review", "Spam", "Filtered (automated)"]
counts = df["folder"].value_counts().to_dict()
counts["All"] = len(df)

chosen = st.radio("Folder", folders, horizontal=True,
                  format_func=lambda f: f"{f} ({counts.get(f, 0)})")
shown = df if chosen == "All" else df[df["folder"] == chosen]

shown = shown.assign(corrected=shown["reviewed"].map({1: "yes", 0: ""}))
st.dataframe(
    shown[["id", "ts", "subject", "category", "folder", "corrected", "final_category",
           "priority", "confidence", "action", "reason"]],
    hide_index=True)

st.subheader("Details")
if shown.empty:
    st.write("Nothing here.")
else:
    pick = st.selectbox("Email id", shown["id"].tolist())
    row = shown[shown["id"] == pick].iloc[0]
    st.write(f"**{row['subject']}**")
    st.text(row["body"])
    exp = json.loads(row["explanation"]) if isinstance(row["explanation"], str) and row["explanation"] else {}
    st.write("Words that pushed the prediction:",
             ", ".join(t for t, _ in exp.get("terms", [])) or "none")
    for cat, sim, subj in exp.get("neighbours", []):
        st.write(f"{sim:.2f} | {cat} | {subj}")
    # ---- route this email to another department
    CATS = load_config(ROOT / "config" / "pipeline.yaml")["categories"] + ["Unknown", "Spam"]
    dept_of = lambda c: "Spam" if c == "Spam" else routing.get(c, "Manual Review")
    current = row["final_category"] if row["reviewed"] and row["final_category"] else row["category"]
    st.markdown("---")
    st.subheader("Route this email")
    dest = st.selectbox("Send this email to", CATS,
                        index=CATS.index(current) if current in CATS else 0,
                        format_func=lambda c: f"{dept_of(c)}  (as {c})",
                        key=f"route_dest_{pick}")
    if st.button("Route email", key=f"route_btn_{pick}"):
        con = sqlite3.connect(DB)
        con.execute("UPDATE results SET reviewed = 1, final_category = ? WHERE id = ?",
                    (dest, int(pick)))
        con.commit()
        con.close()
        st.session_state["flash"] = f"Email {int(pick)} routed to {dept_of(dest)}."
        st.rerun()
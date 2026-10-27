#1_Review_queue.py
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "results.db"

st.set_page_config(page_title="Review queue", layout="wide")
st.title("Review queue")

if not DB.exists():
    st.info("No results yet. Classify an email on the main page first.")
    st.stop()

con = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM results ORDER BY id DESC", con)
con.close()

counts = df.action.value_counts()
c = st.columns(4)
c[0].metric("Auto-routed", int(counts.get("auto_route", 0)))
c[1].metric("Flagged", int(counts.get("route_with_flag", 0)))
c[2].metric("Manual review", int(counts.get("manual_review", 0)))
c[3].metric("Filtered", int(counts.get("filtered", 0)))

view = st.radio("Show", ["Manual review", "Flagged", "Filtered", "All"], horizontal=True)
mapping = {"Manual review": "manual_review", "Flagged": "route_with_flag", "Filtered": "filtered"}
shown = df if view == "All" else df[df.action == mapping[view]]

st.dataframe(
    shown[["id", "ts", "subject", "category", "department", "priority", "confidence", "action", "reason"]],
    hide_index=True)

st.subheader("Details")
if shown.empty:
    st.write("Nothing here.")
else:
    pick = st.selectbox("Email id", shown["id"].tolist())
    row = shown[shown["id"] == pick].iloc[0]
    st.write(f"**{row['subject']}**")
    st.text(row["body"])
    exp = json.loads(row["explanation"]) if row["explanation"] else {}
    st.write("Words that pushed the prediction:",
             ", ".join(t for t, _ in exp.get("terms", [])) or "none")
    for cat, sim, subj in exp.get("neighbours", []):
        st.write(f"{sim:.2f} | {cat} | {subj}")
#2_Batch_upload.py
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.predict import predict
from src.store import save

st.set_page_config(page_title="Batch upload", layout="wide")
st.title("Batch classification")
st.write("Upload a CSV with `subject` and `body` columns.")

f = st.file_uploader("CSV file", type="csv")
if f is not None:
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    missing = {"subject", "body"} - set(df.columns)
    if missing:
        st.error(f"Missing column(s): {', '.join(sorted(missing))}")
        st.stop()
    st.write(f"{len(df)} emails loaded.")
    store = st.checkbox("Also save results to the review queue", value=True)

    if st.button("Classify all", type="primary"):
        rows, bar = [], st.progress(0.0)
        for i, (s, b) in enumerate(zip(df.subject, df.body)):
            r = predict(s, b)
            if store:
                save(s, b, r)
            rows.append(dict(category=r["category"], department=r["department"],
                             priority=r["priority"], confidence=round(float(r["confidence"]), 3),
                             action=r["action"], reason=r.get("reason") or ""))
            bar.progress((i + 1) / len(df))
        st.session_state["batch_out"] = pd.concat([df, pd.DataFrame(rows)], axis=1)

if "batch_out" in st.session_state:
    out = st.session_state["batch_out"]
    st.dataframe(out.drop(columns=["body"]), hide_index=True)
    st.write(out.action.value_counts().to_dict())
    st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "batch_results.csv", "text/csv")
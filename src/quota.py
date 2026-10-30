from pathlib import Path
import streamlit as st
from src.decision import load_config

_CFG = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
MSG = "Session limit reached. Refresh the page to start a new session."

def limit():
    return int(load_config(_CFG).get("limits", {}).get("max_emails_per_session", 10))

def used():
    return st.session_state.get("emails_used", 0)

def remaining():
    return max(0, limit() - used())

def consume(n=1):
    st.session_state["emails_used"] = used() + n

def render(slot):
    slot.metric("Emails used this session", f"{used()} / {limit()}")
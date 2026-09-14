"""Launch with: .venv/bin/python -m streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Bengaluru Bites", page_icon="🍽️", layout="centered")
page = st.navigation([
    st.Page("pages/chat.py", title="Restaurant chat", icon="🍽️", default=True),
    st.Page("pages/1_Review.py", title="Evaluation results", icon="📋", url_path="review"),
    st.Page("pages/handoffs.py", title="Human-review queue", icon="📥", url_path="handoffs"),
])
page.run()

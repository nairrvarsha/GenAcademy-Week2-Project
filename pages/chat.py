"""Run: .venv/bin/python -m streamlit run app.py"""

import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from rag_chat import RestaurantChat
from rag_store import setting




def clear_chat():
    st.session_state.pop("bot", None)
    st.session_state["messages"] = []


def show_answer(reply):
    st.markdown(reply["answer"])
    if reply["sources"]:
        with st.expander(f"Sources · {len(reply['sources'])}"):
            for source in reply["sources"]:
                st.markdown(f"**[{source['label']}] {source['title']}**")
                st.caption(source["chunk_id"])
                st.json(source["content"], expanded=False)
    if reply["status"] == "not_available":
        st.caption("Some information is unavailable. You can ask a follow-up or try another restaurant.")


restaurants = json.loads((ROOT / "data/restaurants.json").read_text())
with st.sidebar:
    st.divider()
    st.markdown("### 🍽️ Bengaluru Bites")
    st.caption(f"Explore {len(restaurants)} restaurant listings")
    st.divider()
    st.markdown("**Narrow your search**")
    locality = st.selectbox("Locality", ["Any locality"] + sorted({r["locality"] for r in restaurants if r.get("locality")}))
    cuisine = st.selectbox("Cuisine", ["Any cuisine"] + sorted({c for r in restaurants for c in (r.get("cuisines") or [])}))
    use_budget = st.checkbox("Set a budget for two")
    budget = st.number_input("Maximum cost for two (₹)", min_value=0, max_value=100000,
                             value=1000, step=100, disabled=not use_budget)
    st.caption("Filters apply to restaurant records. Platform FAQs stay searchable.")
    st.button("New conversation", on_click=clear_chat, use_container_width=True)
    st.divider()
    st.caption("Historical listings. Ratings, hours, and valet are fictional demo values.")

filters = {}
if locality != "Any locality":
    filters["locality"] = locality
if cuisine != "Any cuisine":
    filters["cuisine"] = cuisine
if use_budget:
    filters["max_cost"] = budget
signature = json.dumps(filters, sort_keys=True)
if st.session_state.get("filter_signature", signature) != signature:
    clear_chat()
    st.toast("Filters changed. Started a new conversation.")
st.session_state["filter_signature"] = signature
st.session_state.setdefault("messages", [])

st.title("Find your next bite.")
st.write("Ask about restaurants, cuisines, menus, or how the platform works.")
st.caption("Answers include sources. Menus and prices may be incomplete or outdated.")

ready = True
try:
    setting("NEBIUS_API_KEY")
    setting("PINECONE_API_KEY")
except ValueError:
    ready = False
    st.info("Add NEBIUS_API_KEY and PINECONE_API_KEY to the project's .env file, then refresh.")

suggested = None
if not st.session_state.messages:
    st.markdown("**Try a question**")
    examples = ["What does Benzys cost for two?", "Where can I get Chicken Lollipop?",
                "Why are some menus missing?"]
    for column, example in zip(st.columns(3), examples):
        if column.button(example, disabled=not ready, use_container_width=True):
            suggested = example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            show_answer(message["reply"])

question = st.chat_input("Ask about a restaurant…", max_chars=2000, disabled=not ready) or suggested
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.spinner("Finding relevant listings and preparing your answer…"):
            if "bot" not in st.session_state:
                st.session_state.bot = RestaurantChat()
            reply = st.session_state.bot.ask(question, filters=filters)
    except Exception:
        # Do not display SDK tracebacks or request details in the UI.
        st.error("Couldn't complete the request. Check your connection and API access, then try again.")
    else:
        # Don't retain full debug prompts in the UI history.
        display_reply = {key: reply[key] for key in ("answer", "sources", "status")}
        st.session_state.messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "reply": display_reply},
        ])
        with st.chat_message("assistant"):
            show_answer(display_reply)

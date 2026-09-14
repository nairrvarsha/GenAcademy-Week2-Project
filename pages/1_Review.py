"""Compare saved expected and actual answers without running evaluations."""
import json
from datetime import datetime
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.title("Evaluation results")
st.caption("Compare the expected answer with what the assistant actually returned.")
reports = sorted((ROOT / "evals/runs").glob("*.json"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
if not reports:
    st.info("No saved evaluations yet. Run the eval cell in the notebook first.")
    st.stop()

def report_label(path):
    try:
        date = datetime.strptime(path.stem, "%Y%m%dT%H%M%S%fZ")
        return date.strftime("%d %b %Y, %H:%M:%S UTC")
    except ValueError:
        return path.stem

path = st.selectbox("Evaluation run", reports, format_func=report_label)
report = json.loads(path.read_text())
rows = report["results"]
st.caption(f"{len(rows)} of {report['cases_planned']} results saved")
if not rows:
    st.info("No answers have been saved in this run yet.")
for row in rows:
    st.divider()
    st.subheader(row["case"]["question"])
    setup = row["case"].get("setup_questions", [])
    if setup:
        with st.expander("Earlier conversation"):
            for turn in row.get("turns", [])[:len(setup)]:
                st.write("Question:", turn["question"])
                st.write("Answer:", turn["answer"])
    expected, actual = st.columns(2)
    with expected:
        st.markdown("**Expected answer**")
        st.write(row["case"]["expected_answer"])
    with actual:
        st.markdown("**Actual answer**")
        if row["run_status"] == "completed" and row.get("turns"):
            st.write(row["turns"][-1]["answer"])
        else:
            st.warning("No final answer returned. " + row.get("error_type", "The case did not finish."))

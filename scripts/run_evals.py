"""Run real answer-quality evals; results require human review.

.venv/bin/python scripts/run_evals.py --check-data
.venv/bin/python scripts/run_evals.py --limit 3
.venv/bin/python scripts/run_evals.py
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from langchain_core.documents import Document

from hybrid_retriever import eligible, validate_filters
from prepare_chunks import prepare_chunks
from rag_chat import RestaurantChat
from rag_store import MODEL, ROOT, run_cli

CASES_PATH = ROOT / "evals/cases.json"


def load_cases():
    dataset = json.loads(CASES_PATH.read_text())
    restaurants = json.loads((ROOT / "data/restaurants.json").read_text())
    faqs = json.loads((ROOT / "data/faqs.json").read_text())
    fingerprint = hashlib.sha256(json.dumps({"restaurants": restaurants, "faqs": faqs},
                                           sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if dataset["source_fingerprint"] != fingerprint:
        raise ValueError("Source data changed; review eval expectations and update their source fingerprint.")
    chunk_ids = {chunk["chunk_id"] for chunk in prepare_chunks(restaurants, faqs)}
    chunk_ids.add("restaurants:structured-query")
    cases = dataset["cases"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Eval IDs must be unique.")
    for case in cases:
        validate_filters(case["filters"])
        for group in case["expected_source_groups"]:
            if not group or not set(group).issubset(chunk_ids):
                raise ValueError(f"Unknown expected source in {case['id']}.")
        for pattern in case["required_patterns"] + case["forbidden_patterns"]:
            re.compile(pattern)
    return cases


def screen_answer(case, reply):
    """Cheap, explicit screening checks, NOT a semantic correctness judge."""
    retrieved = {source["chunk_id"] for source in reply["retrieved_context"]}
    cited = {source["chunk_id"] for source in reply["sources"]}
    groups = case["expected_source_groups"]
    checks = {
        "expected_status": reply["status"] in case["acceptable_statuses"],
        "required_facts_or_phrases": all(re.search(p, reply["answer"], re.I) for p in case["required_patterns"]),
        "no_forbidden_phrases": not any(re.search(p, reply["answer"], re.I) for p in case["forbidden_patterns"]),
        "expected_source_retrieved": all(set(g) & retrieved for g in groups) if groups else None,
        "expected_source_cited": all(set(g) & cited for g in groups) if groups else None,
        "citations_belong_to_retrieved_context": cited.issubset(retrieved),
        "retrieved_restaurants_match_filters": all(
            eligible(Document(page_content="", metadata=s["metadata"]), case["filters"])
            for s in reply["retrieved_context"]
        ),
    }
    assessment = reply.get("support_assessment")
    if assessment is not None:
        expected = case["acceptable_statuses"]
        if expected == ["answered"]:
            checks["expected_handoff_decision"] = not assessment["needs_human"]
        elif "answered" not in expected:
            checks["expected_handoff_decision"] = assessment["needs_human"]
    return checks


def summarize(report):
    rows = report["results"]
    complete = [r for r in rows if r["run_status"] == "completed"]
    with_refs = [r for r in complete if r["checks"]["expected_source_retrieved"] is not None]
    reviewed = [r for r in complete if all(r["human_review"][k] is not None
                for k in ("correctness", "grounding", "appropriate_behavior"))]
    resolved = [r for r in reviewed if not r["case"]["setup_questions"]
                and r["human_review"]["resolved_without_handoff"] is not None]
    def rate(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else None
    return {
        "cases_planned": report["cases_planned"], "cases_recorded": len(rows),
        "api_errors": len(rows) - len(complete),
        "automatic_screen_pass_rate": rate(sum(r["automatic_checks_pass"] for r in complete), len(rows)),
        "expected_source_hit_rate": rate(sum(r["checks"]["expected_source_retrieved"] for r in with_refs), len(with_refs)),
        "human_reviewed_cases": len(reviewed),
        "human_review_pass_rate": rate(sum(all(r["human_review"][k] for k in
            ("correctness", "grounding", "appropriate_behavior")) for r in reviewed), len(reviewed)),
        "reviewed_single_turn_resolution_rate": rate(sum(
            r["human_review"]["resolved_without_handoff"] and all(r["human_review"][k] for k in
            ("correctness", "grounding", "appropriate_behavior")) for r in resolved), len(resolved)),
    }


def save_report(report):
    report["summary"] = summarize(report)
    path = Path(report["report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_evals(limit=None, bot=None, output_dir=None):
    cases = load_cases()
    if limit is not None:
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        cases = cases[:limit]
    if bot is None:
        missing = [name for name in ("data/chunks.json", "data/pinecone_index.json") if not (ROOT / name).exists()]
        if missing:
            raise ValueError("Run notebook sections 1–4 first. Missing: " + ", ".join(missing))
        bot = RestaurantChat(persist_handoffs=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "evals/runs"
    pipeline_files = ["scripts/rag_chat.py", "scripts/hybrid_retriever.py", "scripts/escalation.py", "scripts/evidence_check.py", "prompts/evidence_check.txt", "scripts/structured_search.py", "prompts/query_planner.txt", "prompts/restaurant_system.txt"]
    report = {
        "run_id": run_id, "report_path": str(output_dir / f"{run_id}.json"),
        "cases_planned": len(cases), "run_status": "running",
        "chat_model": str(getattr(bot.llm, "model_name", "unknown")), "embedding_model": MODEL,
        "eval_dataset_hash": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        "pipeline_hash": hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in pipeline_files)).hexdigest(),
        "results": [],
    }
    save_report(report)
    for case in cases:
        bot.reset()  # Each case is isolated; only its own setup questions supply history.
        started = time.perf_counter()
        row = {"case": case, "turns": [], "human_review": {
            "correctness": None, "grounding": None, "appropriate_behavior": None,
            "resolved_without_handoff": None, "notes": "",
        }}
        try:
            for question in case["setup_questions"] + [case["question"]]:
                reply = bot.ask(question, filters=case["filters"])
                row["turns"].append({"question": question, "answer": reply["answer"],
                                     "status": reply["status"], "sources": reply["sources"],
                                     "retrieved_context": reply["retrieved_context"],
                                     "retrieval_query": reply["retrieval_query"],
                                     "graph_steps": reply.get("graph_steps", []),
                                     "support_assessment": reply.get("support_assessment")})
            row["checks"] = screen_answer(case, reply)
            row["automatic_checks_pass"] = all(v for v in row["checks"].values() if v is not None)
            row["run_status"] = "completed"
        except KeyboardInterrupt:
            report['run_status'] = 'interrupted'
            save_report(report)
            print('Interrupted; returning completed results:', report['report_path'])
            return report
        except Exception as error:
            row.update(run_status="error", error_type=type(error).__name__, checks={}, automatic_checks_pass=False)
        row["seconds"] = round(time.perf_counter() - started, 2)
        report["results"].append(row)
        save_report(report)  # Preserve progress even if a later API call fails.
        label = "SCREEN PASS" if row["automatic_checks_pass"] else ("API ERROR" if row["run_status"] == "error" else "NEEDS REVIEW")
        print(f"{len(report['results'])}/{len(cases)} {case['id']}: {label}", flush=True)
    report["run_status"] = "completed_with_errors" if any(r["run_status"] == "error" for r in report["results"]) else "completed"
    save_report(report)
    return report


def review_rows(report):
    return [{"id": row["case"]["id"], "question": row["case"]["question"],
             "expected": row["case"]["expected_answer"],
             "actual": row["turns"][-1]["answer"] if row["run_status"] == "completed" else "API ERROR: " + row["error_type"],
             "automatic_screen_pass": row["automatic_checks_pass"],
             "source_hit": row["checks"].get("expected_source_retrieved"),
             "seconds": row["seconds"]} for row in report["results"]]


def record_review(report_path, case_id, *, correctness, grounding, appropriate_behavior,
                  resolved_without_handoff, notes=""):
    ratings = dict(correctness=correctness, grounding=grounding, appropriate_behavior=appropriate_behavior,
                   resolved_without_handoff=resolved_without_handoff)
    if not all(isinstance(value, bool) for value in ratings.values()):
        raise ValueError("Review ratings must be True or False.")
    report = json.loads(Path(report_path).read_text())
    row = next((r for r in report["results"] if r["case"]["id"] == case_id), None)
    if row is None or row["run_status"] != "completed":
        raise ValueError("Review a completed case in this report.")
    row["human_review"] = {**ratings, "notes": notes}
    report["report_path"] = str(Path(report_path))
    save_report(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.check_data:
        print(f"Validated {len(load_cases())} eval cases and their expected source IDs. No API calls.")
        return
    report = run_evals(limit=args.limit)
    print(json.dumps(report["summary"], indent=2))
    print("Review answers and sources in:", report["report_path"])


if __name__ == "__main__":
    run_cli(main)

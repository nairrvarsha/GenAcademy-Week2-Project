# Answer-quality evals

`cases.json` contains 22 questions with expected behavior, source IDs, and screening
checks grounded in the current restaurant/FAQ data. Cases cover facts, menus,
synthetic-field disclosure, filtered recommendations, unavailable data, unsupported
requests, a follow-up, and an instruction to claim a fake listing update.

Complete notebook sections 1–4 before a real eval run. Then use notebook section 9,
or run:

```sh
.venv/bin/python scripts/run_evals.py --check-data
.venv/bin/python scripts/run_evals.py --limit 3
.venv/bin/python scripts/run_evals.py
```

Only the last two commands use APIs/credits. The full run evaluates 22 cases; the
follow-up case also asks its setup question. Cases start with separate histories.
Results are saved after every case in `evals/runs/<run-id>.json`, including actual
answers, complete retrieved/cited content, source checks, latency, and errors.
The file also records model, prompt/code, and eval-dataset fingerprints.

## How to assess a result

Automatic screening checks expected status, required/forbidden phrases, expected
source retrieval/citation, and filters. These checks are deliberately transparent
heuristics. They can flag good paraphrases or miss incorrect claims. A screen pass
is NOT answer correctness and does not prove citation entailment.

For each saved result, compare `case.expected_answer` with the last turn's answer
and its cited records. Assign four human-review booleans:

- `correctness`: all facts and requested conditions are right; no invented details.
- `grounding`: the cited records support the factual claims.
- `appropriate_behavior`: missing data, ambiguity, demo fields, and unsupported
  operations are handled as the expected answer describes.
- `resolved_without_handoff`: the user's information need is resolved without
  human intervention. Saying information is unavailable is not automatically resolution.

Use `record_review()` from the notebook, or edit the human-review values in a saved
report and call `summarize(report)`. Record notes explaining failures. Initially
all human reviews are null: no human-reviewed accuracy is claimed until reviewed.

The summary reports automatic-screen rate, expected-source hit rate, reviewed
answer pass rate, and a reviewed single-turn resolution rate. The last metric is
an evaluation proxy, not a measured production first-contact-resolution rate.
Follow-up cases are excluded from that single-turn resolution metric.

The eval source fingerprint prevents silently using stale expectations after data
changes. Review and update expected answers/source IDs before updating that fingerprint.
No LLM judge or LangSmith service is configured yet.

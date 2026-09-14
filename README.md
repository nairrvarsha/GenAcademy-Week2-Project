# RAG Walkthrough

Discover restaurants in Bengaluru using LangChain, Nebius, and Pinecone.
The dataset contains 200 historical restaurant listings and 12 project FAQs.
Ratings, opening hours, and valet availability are fictional demo values.

## Run the notebook

Open `RAG_Walkthrough.ipynb`, select the `.venv` Python kernel, and execute the
sections in order:

1. Project setup — imports functions; no API calls.
2. Load data and create chunks — creates `data/chunks.json`.
3. Embed the data — calls Nebius and keeps vectors in notebook memory.
4. Store embeddings in Pinecone — uploads those vectors without re-embedding and
   creates `data/pinecone_index.json`. It creates the Pinecone index if missing, or reuses the existing index.
5. Ask a question and retrieve context — combines BM25 and semantic search.
6. Build the prompt — system instructions, question, and retrieved JSON.
7. Generate an answer — calls the Nebius chat model and checks citation labels.
8. Review the sources — displays the cited records.
9. Evaluate the assistant — runs the 20 answer-quality cases and saves results.

To ask another standalone question, rerun sections 5–8. For a full restart after
kernel memory is cleared, rerun sections 1–4 first. Section 3 uses API credits each
time; vectors are not saved locally. The existing remote index is not deleted by
local file cleanup.

`data/chunks.json` and `data/pinecone_index.json` are recreated by sections 2 and 4.
Section 9 creates evaluation reports under `evals/runs/`. The notebook cannot
recreate restaurant/FAQ source data, eval definitions, API keys, prompts, or Python code.

## Chat interface

After notebook section 4 completes:

```sh
.venv/bin/python -m streamlit run app.py
```

The UI supports follow-ups, expandable sources, and exact locality, cuisine,
budget-for-two, and valet filters. FAQs remain searchable with restaurant filters.
History is kept in memory per browser session; changing filters or starting a new
conversation clears it. Notebook questions use empty history.

## Project files

- `data/restaurants.json`, `data/faqs.json`: source data; keep these.
- `prompts/restaurant_system.txt`: editable system instructions.
- `scripts/prepare_chunks.py`: record-based JSON chunking.
- `scripts/rag_store.py`: shared LangChain connections and Documents.
- `scripts/push_vectors.py`: stores precomputed notebook vectors using Pinecone's SDK.
- `scripts/upload_to_pinecone.py`: optional terminal uploader; creates an index if
  missing and resumes uploads while reusing matching stored vectors. Kept because
  those operations are not covered by the notebook upload cell.
- `scripts/hybrid_retriever.py`: BM25 + Pinecone, merged with reciprocal rank fusion.
- `scripts/rag_chat.py`: shared prompt, generation, citation checks, and chat history.
- `scripts/run_evals.py`: runs evals and saves results for review.
- `evals/cases.json`, `evals/README.md`: eval definitions and review instructions.
- `.streamlit/config.toml`: UI theme and configuration.
- `.env`: private API keys; keep it out of Git.
- `.venv/`: installed Python environment; keep it to run the notebook.

## Setup on a new machine

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Create `.env` with `NEBIUS_API_KEY` and `PINECONE_API_KEY`. The optional
`NEBIUS_CHAT_MODEL` overrides the default `Qwen/Qwen3-30B-A3B-Instruct-2507`.
Embeddings use `Qwen/Qwen3-Embedding-8B` with 4096 dimensions.
Notebook section 4 creates a missing Pinecone index and waits for it to be ready.
Existing compatible indexes are reused; incompatible indexes are left untouched.

Hybrid search takes up to 20 candidates from each method, merges ranks, and keeps
five chunks. RRF and cosine scores are not confidence percentages. Citation-label
validation does not guarantee factual accuracy. Hard filters come from the sidebar;
automatic extraction of all constraints from questions is not implemented.
A calibrated confidence policy and live human support integration remain future work. LangGraph now orchestrates the online conversation.

## Answer-quality evaluations

`evals/cases.json` defines 22 questions and expected behavior. Notebook section 9
previews them and runs the real assistant. Alternatively:

```sh
.venv/bin/python scripts/run_evals.py --check-data
.venv/bin/python scripts/run_evals.py --limit 3
```

Results go to `evals/runs/`. Source/phrase checks are automated; semantic correctness
and citation support require human review. See `evals/README.md` for the rubric.

Simple count/list questions use `scripts/structured_search.py` to query all restaurant records by locality (plus explicit app filters). Counts describe this dataset, not every restaurant in Bengaluru. Unsupported aggregate constraints ask for clarification. Descriptive questions continue to use hybrid retrieval.

## Human review and evaluation report

Open Evaluation results in Streamlit to compare expected and actual answers. Record human judgments with `record_review()` in the notebook and export a Markdown report with its final cell. Unknown judgments stay unreviewed. The report shows review coverage, human-reviewed faithfulness, and an offline first-contact resolution proxy; it does not claim production resolution rates.

Responses without sufficient evidence enter `data/human_review_queue/` during app conversations. Reviewers can save a resolution on the Human-review queue page. This is a local demo workflow, not delivery to a staffed helpdesk. The fallback checks status/citations and uses an LLM evidence verifier for generated RAG answers. This is not calibrated confidence. Eval runs record the fallback decision without creating real queue items.

Interrupted eval runs return completed results. Logs distinguish NEEDS REVIEW from API ERROR. The report includes per-case failures and reviewer notes. Proposed demo goals are 90% faithfulness and 80% offline resolution; calibration and customer validation remain future work.

Query routing uses an LLM plan defined in `prompts/query_planner.txt`, validated with Pydantic. Python applies supported filters and calculates counts; it never executes generated code. Each question adds a planning model call. Unsupported aggregate constraints ask for clarification.

## LangGraph workflow

`RestaurantChat.graph` runs resolve_followup → plan_query → retrieve (RAG only) → answer → check_evidence → handoff (when needed). Counts and lists skip retrieval and the generative evidence verifier because Python computes them from a validated plan. A bad planner decision is still possible and must be evaluated.

RAG answers receive an extra model call against their cited sources. Unsupported answers and verifier failures are withheld and routed to the local queue. The verifier itself can make mistakes; it is not a calibrated confidence score. History remains the last three exchanges in memory, with no checkpointer or durable conversation resume. Human queue resolution is a local workflow, not a graph interrupt/resume or external notification.

The notebook includes a complete graph-run cell after the individual component walkthrough. Streamlit and evals call the same graph automatically. Changing code may require starting a new Streamlit conversation.

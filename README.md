# Postmortem RAG: Quality-Gated Ingestion on Airflow 3

Reference implementation for the Airflow Summit 2026

**Scenario:** incident postmortems are the source corpus. Engineers query
the resulting RAG index to understand past incidents, root causes, and
remediations instead of relying on tribal knowledge or a wiki search that
misses the relevant doc. The pipeline's job is to keep that index both
fresh and *correct* -- a green DAG run is necessary but not sufficient.

## Layout

```
dags/postmortem_rag_pipeline.py   Airflow 3 DAG: TaskFlow + Assets + quality gates + guardrails
dags/postmortem_query_pipeline.py Airflow 3 DAG: answer a question against PROD, with query-time guardrails
src/ingest.py                     chunking, PII/secret redaction, embedding, Chroma staging/prod collections
src/evaluate.py                   the structural + RAGAS quality gates, plus the guardrail gates
src/guardrails.py                 ingest-time PII/secret scanner + query-time LLM guardrail models/prompts
src/query.py                      thin CLI wrapper that triggers postmortem_query_pipeline
streamlit_app.py                  local UI: triggers postmortem_query_pipeline over the REST API, polls for the answer
data/postmortems/*.md             5 synthetic sample postmortems (source corpus)
data/eval_dataset.jsonl           golden question set used by the RAGAS gate
tests/run_smoketest.py            offline logic smoke test (see below)
tests/fakes/                      fake chromadb/openai/datasets/ragas/airflow.sdk used by the smoke test
```

### Apache Airflow's common.ai provider

Generation and both LLM-judged guardrail checks run through
[`apache-airflow-providers-common-ai`](https://airflow.apache.org/docs/apache-airflow-providers-common-ai/stable/)
(built on pydantic-ai) instead of a raw client call in a plain Python
function. That's a deliberate architectural choice, not just a dependency
swap: `@task.llm` only produces a real, observable, retryable task instance
when it's part of a DAG's task graph, so every LLM call in this repo --
generation, input-safety judging, groundedness judging -- now shows up as
its own task in the Airflow UI with its own logs and retries, including the
mapped one-per-golden-question tasks in the ingestion DAG.

This project runs on **open-source models hosted by [Hugging Face's
Inference Providers](https://huggingface.co/docs/inference-providers)** --
you need an [HF API token](https://huggingface.co/settings/tokens), but no
GPU of your own and no local model server. Generation and the input-safety
guardrail run on `pydanticai_default` (default
`meta-llama/Llama-3.3-70B-Instruct:novita`); the groundedness guardrail
runs on its own connection, `pydanticai_groundedness` (same default model,
but kept separate since claim-attribution judging benefits from a
strong/large model independently of whatever generation model you pick).
The RAGAS eval judge (`PM_RAG_GENERATION_MODEL`) uses the same default
model directly via the `openai` client. Embeddings use a separate hosted
embedding model (default `sentence-transformers/all-MiniLM-L6-v2`) via
`huggingface_hub`'s `InferenceClient`.

Generate a token at https://huggingface.co/settings/tokens, then:

```bash
export HF_TOKEN=hf_your_token_here
export HF_BASE_URL=https://router.huggingface.co/v1
export AIRFLOW_CONN_PYDANTICAI_DEFAULT='{"conn_type": "pydanticai", "host": "'"$HF_BASE_URL"'", "password": "'"$HF_TOKEN"'", "extra": {"model": "openai:meta-llama/Llama-3.3-70B-Instruct:novita"}}'
export AIRFLOW_CONN_PYDANTICAI_GROUNDEDNESS='{"conn_type": "pydanticai", "host": "'"$HF_BASE_URL"'", "password": "'"$HF_TOKEN"'", "extra": {"model": "openai:meta-llama/Llama-3.3-70B-Instruct:novita"}}'
```

Swap `meta-llama/Llama-3.3-70B-Instruct:novita` for any other chat model
Inference Providers serves (model ids take the form
`org/model:provider`, e.g. `Qwen/Qwen2.5-72B-Instruct:novita`) by changing
the connection's `model` (and `PM_RAG_GENERATION_MODEL` to match, for the
RAGAS judge), and swap the embedding model via the
`PM_RAG_EMBEDDING_MODEL` env var -- no code changes needed. Structured
guardrail output (`GROUNDEDNESS_OUTPUT_TYPE`/`INPUT_SAFETY_OUTPUT_TYPE` in
`src/guardrails.py`) uses `PromptedOutput` rather than native structured
output, since not every Inference Providers backend supports
function-calling or `response_format: json_schema` -- worth rechecking if
you swap to a provider/model that does. See
[huggingface.co/docs/inference-providers/pricing](https://huggingface.co/docs/inference-providers/pricing)
for current rates; every account gets some free monthly credit, and HF
passes through the underlying provider's price with no markup.

### Guardrails vs. quality gates

RAGAS (`check_retrieval_quality`) scores answer/retrieval quality
*continuously* against a floor -- a slightly-below-floor run and a
wildly-below-floor run both just "fail the gate." Guardrails are hard
pass/fail checks that don't average away, at two stages:

- **Ingest-time** (`src/guardrails.py::scan_and_redact`, deterministic, no
  LLM call): every source document is scanned for emails/tokens (soft --
  redacted, ingestion continues) and cloud provider access keys / private
  key blocks (hard -- also redacted, but `check_pii_hard_block` blocks
  promotion to prod regardless of RAGAS scores until a human reviews and
  re-ingests a cleaned copy). See
  `data/postmortems/INC-2350-support-ticket-leak.md` for the fixture this
  is built to catch.
- **Query-time** (`dags/postmortem_query_pipeline.py`, LLM-judged via
  `@task.llm` with structured Pydantic output): `check_input_safety`
  refuses prompt-injection / off-topic / action-seeking questions before
  any retrieval happens, and `check_groundedness` blocks a generated
  answer whose claims aren't supported by the retrieved context --
  including the case where the model would otherwise follow an
  instruction embedded inside a postmortem document instead of just
  reporting on it (also exercised by `INC-2350`'s "ignore all previous
  instructions" sentence).

### Asking a question

`@task.llm` only works inside a DAG's task graph, so answering a question
is now an Airflow DAG run rather than an in-process CLI call:

```bash
python -m src.query "What caused the checkout latency spike?"
# equivalent to:
airflow dags trigger postmortem_query_pipeline --conf '{"question": "..."}'
```

Check the run's task logs in the Grid UI: `deliver_answer` if both
guardrails passed, `refuse` if the input-safety guardrail blocked the
question, or `block_answer` if the groundedness guardrail blocked the
generated answer.

### Asking a question from a local UI

`streamlit_app.py` is a small local UI for the same DAG -- no Airflow UI, no
CLI, and (unlike a Slack integration) no public exposure needed, since it
only ever talks to `localhost`. It triggers `postmortem_query_pipeline` over
Airflow's REST API with a client-chosen `dag_run_id`, then polls
`include/query_results.db` (written by `_record_result` in
`dags/postmortem_query_pipeline.py`, keyed by that same run id) until the
answer, refusal, or block message shows up.

```bash
pip install streamlit requests
streamlit run streamlit_app.py
```

By default it talks to `http://localhost:8080` with the `admin`/`admin`
credentials from `airflow.cfg`'s `simple_auth_manager_users` -- override with
the `AIRFLOW_BASE_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` env vars if
your local Airflow API server is mapped to a different port or uses
different credentials (check with `docker port <project>-api-server-1`).

The Chroma collection names in `src/ingest.py` (`postmortem_index_staging`,
`postmortem_index_prod`) match the Airflow `Asset` names in the DAG file
one-to-one on purpose -- an Asset event and the physical index it
describes should never drift into different naming schemes, which is
exactly the kind of silent mismatch this whole pipeline exists to prevent.



| Slide | Claim | Backed by |
|---|---|---|
| 4 | The four failure modes | `check_chunking_regression`, `check_embedding_model_drift`, `check_partial_reindex`, `check_retrieval_quality` in `src/evaluate.py` |
| 5 | Staging → gates → promote/block architecture | `dags/postmortem_rag_pipeline.py` task graph |
| 6 | RAGAS floors: faithfulness 0.85, context precision/recall 0.75 | `RAGAS_FLOORS` in `src/evaluate.py` |
| 7 | Asset names `postmortem_index_staging` / `postmortem_index_prod` | `Asset(...)` calls in the DAG **and** `STAGING_COLLECTION` / `PROD_COLLECTION` in `src/ingest.py` (kept identical on purpose -- see above) |
| 8 | `rerun_with_latest_version=False` | passed directly to the `DAG(...)` constructor in `dags/postmortem_rag_pipeline.py` |
| 9 | Demo task sequence (`build_staging_index` → `publish_staging_asset` → `evaluate_quality_gates` → `promote_to_prod`/`block_promotion`) | task names in `dags/postmortem_rag_pipeline.py`, unchanged |
| 13 | Repo contents | `dags/`, `src/`, `tests/run_smoketest.py` |


## The failure modes and guardrails, and where each is caught

| Check | Where it's caught | How |
|---|---|---|
| Chunking regressions | `check_chunking_regression` | compares chunk count / avg chunk size to the last **passing** run; fails if drift exceeds threshold |
| Embedding model drift | `check_embedding_model_drift` | compares the embedding model name recorded in this run's manifest to the last passing run's |
| Partial re-index states | `check_partial_reindex` | confirms every source document present on disk made it into the staging manifest |
| PII/secret hard block (guardrail) | `check_pii_hard_block` | fails if any source document contained a hard-severity secret (cloud access key, private key block), even after redaction |
| Retrieval quality decay | `check_retrieval_quality` | runs the golden question set's common.ai-generated answers through RAGAS (`faithfulness`, `context_precision`, `context_recall`) against fixed floors |
| Ungrounded answers (guardrail) | `check_answer_guardrails` | fails if any golden-set answer was judged ungrounded by the `@task.llm` groundedness check |

All six run against the **staging** Chroma collection. Production is only
overwritten by `promote_to_prod`, which is only reached if every check
passes. A failed check raises in `block_promotion` -- the DAG run fails
loudly, and prod keeps serving the last known-good index.

## Offline logic smoke test (no API keys / network needed)

`tests/run_smoketest.py` runs the real ingestion, quality-gate, and
guardrail logic end-to-end against hand-written fake `chromadb` /
`openai` / `datasets` / `ragas` / `airflow.sdk` modules (in
`tests/fakes/`). It proves the Python logic itself — chunking, PII
redaction, manifest diffing, all structural and guardrail gates (including
their *failure* paths), the RAGAS-gate wiring, promotion, the query CLI's
trigger command, and both DAG files' task wiring (including dynamic task
mapping and `@task.llm`) — is free of bugs, without needing network access
or real credentials. It does **not** validate real embedding/retrieval
quality, real RAGAS scores, or real LLM guardrail judgments — run the real
pipeline for that.

```bash
python3 tests/run_smoketest.py
```

Exits 0 with a `PASS`/`FAIL` line per check; useful as a pre-flight before
a live demo, or as a starting point for real unit tests once you're
plugging in an actual corpus.

## Running locally

```bash
pip install -r requirements.txt

# Generate a token at https://huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here
export HF_BASE_URL=https://router.huggingface.co/v1
export AIRFLOW_CONN_PYDANTICAI_DEFAULT='{"conn_type": "pydanticai", "host": "'"$HF_BASE_URL"'", "password": "'"$HF_TOKEN"'", "extra": {"model": "openai:meta-llama/Llama-3.3-70B-Instruct:novita"}}'
export AIRFLOW_CONN_PYDANTICAI_GROUNDEDNESS='{"conn_type": "pydanticai", "host": "'"$HF_BASE_URL"'", "password": "'"$HF_TOKEN"'", "extra": {"model": "openai:meta-llama/Llama-3.3-70B-Instruct:novita"}}'

# Point Airflow at this project's dags/ folder, then trigger
# postmortem_rag_pipeline from the UI or:
airflow dags trigger postmortem_rag_pipeline

# Once promoted, ask it something:
python -m src.query "What caused the checkout latency spike?"
```

## Adapting this for a real postmortem corpus

- Swap `data/postmortems/*.md` for an export of your actual incident
  tracker (PagerDuty, Jira, Rootly, etc.) -- one document per incident is
  the right granularity; don't concatenate a quarter's postmortems into
  one file, or chunking will mix unrelated incidents into one chunk.
- Regenerate `data/eval_dataset.jsonl` from your own corpus. RAGAS's
  reference-free metrics (faithfulness, answer relevancy) don't need a
  ground truth; `context_recall` does, so keep a small hand-written golden
  set even as the corpus grows.
- Replace `schedule=None` with an Asset-based schedule keyed off your
  incident tracker's webhook, or a lightweight sensor, so a new postmortem
  triggers a refresh automatically rather than waiting for a nightly cron.
- The RAGAS floors in `src/evaluate.py` (0.85 faithfulness, 0.75 precision
  and recall) are a reasonable starting point, not a universal constant --
  run the gate against your own historical "known good" index first to see
  where your baseline actually sits, then set floors slightly below that.
- `SENSITIVE_PATTERNS` in `src/guardrails.py` is a starting set (emails,
  bearer tokens, cloud access keys, private key blocks) -- extend it with
  your own org's secret formats, and tune the two guardrail system prompts
  (`INPUT_GUARDRAIL_SYSTEM_PROMPT`, `GROUNDEDNESS_GUARDRAIL_SYSTEM_PROMPT`)
  against real questions/answers from your corpus before trusting them to
  block automatically rather than just flag for review.

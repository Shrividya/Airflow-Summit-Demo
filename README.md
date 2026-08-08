# Postmortem RAG: Quality-Gated Ingestion on Airflow 3

Reference implementation for Airflow Summit 2026. Incident postmortems are
ingested into a RAG index that engineers query for root causes and
remediations. The pipeline keeps that index fresh *and* correct — a green
DAG run alone isn't enough, so every promotion to prod passes through
quality gates, guardrails, and (on failure) human review.


## How it works

**Ingest** (`postmortem_rag_pipeline`): chunk + embed postmortems into a
staging Chroma collection, generate answers to a golden question set, then
run quality gates against staging. All gates pass → promote to prod. Any
gate fails → route to a human-approval task (`ApprovalOperator`) instead of
retrying blindly; a PII/secret leak also pages Slack. Either way, the run
still fails loudly and prod keeps serving the last known-good index.

**Query** (`postmortem_query_pipeline`): a question is checked for
safety/prompt-injection before retrieval, an answer is generated from
retrieved context, and the answer is checked for groundedness before
delivery. Triggered via `python -m src.query "..."` or `streamlit_app.py`.

Both DAGs run LLM calls through `apache-airflow-providers-common-ai`
(`@task.llm`), so generation and every guardrail check show up as its own
observable, retryable task in the Airflow UI — not a black-box function
call.

| Gate/guardrail | Catches |
|---|---|
| `check_chunking_regression` | chunk count/size drift vs. last passing run |
| `check_embedding_model_drift` | embedding model changed since last passing run |
| `check_partial_reindex` | source docs missing from the staging manifest |
| `check_pii_hard_block` | hard-severity secrets (access keys, private keys) → routes to `security_incident_review` + Slack |
| `check_retrieval_quality` | RAGAS faithfulness/precision/recall below floor |
| `check_answer_guardrails` | golden-set answers judged ungrounded |
| `check_input_safety` (query-time) | prompt injection / off-topic / action-seeking questions |
| `check_groundedness` (query-time) | answer claims not supported by retrieved context |

## Setup

Models are hosted via [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers)
— no GPU or local model server needed, just an [HF token](https://huggingface.co/settings/tokens).

```bash
pip install -r requirements.txt

export HF_TOKEN=hf_your_token_here
export HF_BASE_URL=https://router.huggingface.co/v1
export AIRFLOW_CONN_PYDANTICAI_DEFAULT='{"conn_type": "pydanticai", "host": "'"$HF_BASE_URL"'", "password": "'"$HF_TOKEN"'", "extra": {"model": "openai-chat:meta-llama/Llama-3.3-70B-Instruct:novita"}}'
export AIRFLOW_CONN_PYDANTICAI_GROUNDEDNESS='{"conn_type": "pydanticai", "host": "'"$HF_BASE_URL"'", "password": "'"$HF_TOKEN"'", "extra": {"model": "openai-chat:meta-llama/Llama-3.3-70B-Instruct:novita"}}'

# optional — enables the Slack alert on a PII/secret hard-block
export AIRFLOW_CONN_SLACK_DEFAULT='{"conn_type": "slackwebhook", "password": "https://hooks.slack.com/services/your/webhook/path"}'
```

Swap models by changing the connection's `model` (and `PM_RAG_GENERATION_MODEL`
to match, for the RAGAS judge) or the embedding model via `PM_RAG_EMBEDDING_MODEL`
— no code changes needed. `airflow_settings.yaml` has the equivalent local
Astro CLI connection setup.

## Running

```bash
airflow dags trigger postmortem_rag_pipeline

python -m src.query "What caused the checkout latency spike?"
# or:
streamlit run streamlit_app.py
```

## Adapting for a real corpus

- Swap `data/postmortems/*.md` for an export of your incident tracker —
  one document per incident, so chunking doesn't mix unrelated incidents.
- Regenerate `data/eval_dataset.jsonl` from your own corpus; keep a small
  hand-written golden set since RAGAS's `context_recall` needs ground truth.
- Replace `schedule=None` with an Asset-based schedule tied to your
  tracker's webhook.
- Re-baseline the RAGAS floors in `src/evaluate.py` against your own
  "known good" index rather than trusting the defaults.
- Extend `SENSITIVE_PATTERNS` in `src/guardrails.py` with your org's secret
  formats, and tune both guardrail prompts against real Q&A before trusting
  them to block automatically.
- Point `security_incident_review`'s Slack alert and response timeout at
  your real on-call/incident-response process.

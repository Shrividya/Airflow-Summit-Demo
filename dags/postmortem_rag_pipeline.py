"""Ingest incident postmortems into a quality-gated, guardrailed RAG index.
Uses TaskFlow, Assets for the staging/prod indices, `rerun_with_latest_version=False`
so reruns replay the original chunking/embedding code, and common-ai's
`@task.llm` (mapped one task per golden question) for generation and the
groundedness guardrail. Gate checks against STAGING before promotion live
in src/evaluate.py: chunking regression, embedding model drift, partial
re-index, PII/secret hard block, RAGAS retrieval quality, and ungrounded
answers.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

from airflow.sdk import DAG, Asset, task
from airflow.providers.standard.operators.hitl import ApprovalOperator
from airflow.providers.slack.notifications.slack_webhook import SlackWebhookNotifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.guardrails import (  # noqa: E402
    CACHED_INSTRUCTIONS_SETTINGS,
    GROUNDEDNESS_GUARDRAIL_SYSTEM_PROMPT,
    GroundednessVerdict,
)

SOURCE_DIR = os.environ.get("PM_RAG_SOURCE_DIR", "/usr/local/airflow/data/postmortems")
EVAL_DATASET_PATH = os.environ.get("PM_RAG_EVAL_DATASET", "/usr/local/airflow/data/eval_dataset.jsonl")
LLM_CONN_ID = os.environ.get("PM_RAG_LLM_CONN_ID", "pydanticai_default")

GENERATION_SYSTEM_PROMPT = (
    "Answer the question using ONLY the postmortem excerpts provided. "
    "If the excerpts don't contain the answer, say so."
)

postmortem_index_staging = Asset("postmortem_index_staging")
postmortem_index_prod = Asset("postmortem_index_prod")

with DAG(
    dag_id="postmortem_rag_pipeline",
    description="Ingest incident postmortems into a quality-gated, guardrailed RAG index",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["rag", "quality-gates", "guardrails", "common-ai", "reference-architecture"],
    rerun_with_latest_version=False,
):

    @task(
        retries=4,
        retry_delay=timedelta(minutes=1),
        retry_exponential_backoff=True,
        max_retry_delay=timedelta(minutes=15),
    )
    def build_staging_index() -> dict:
        """Chunk + embed every postmortem into the STAGING collection."""
        from src.ingest import build_staging_index as _build

        manifest = _build(SOURCE_DIR)
        return manifest.__dict__

    @task(outlets=[postmortem_index_staging])
    def publish_staging_asset(manifest: dict) -> dict:
        return manifest

    @task
    def retrieve_eval_contexts(manifest: dict) -> list[dict]:
        from src.ingest import retrieve, STAGING_COLLECTION

        rows = [json.loads(line) for line in open(EVAL_DATASET_PATH) if line.strip()]
        items = []
        for row in rows:
            hits = retrieve(row["question"], collection_name=STAGING_COLLECTION, k=4)
            items.append({
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "contexts": [h["text"] for h in hits],
            })
        return items

    @task.llm(llm_conn_id=LLM_CONN_ID, system_prompt=GENERATION_SYSTEM_PROMPT, agent_params=CACHED_INSTRUCTIONS_SETTINGS)
    def generate_eval_answer(item: dict) -> str:
        context_block = "\n\n---\n\n".join(item["contexts"])
        return f"Excerpts:\n{context_block}\n\nQuestion: {item['question']}\nAnswer:"

    @task
    def zip_generated_answers(items: list[dict], answers: list[str]) -> list[dict]:
        return [{**item, "answer": answer} for item, answer in zip(items, answers)]

    @task.llm(
        llm_conn_id=LLM_CONN_ID,
        system_prompt=GROUNDEDNESS_GUARDRAIL_SYSTEM_PROMPT,
        output_type=GroundednessVerdict,
        agent_params=CACHED_INSTRUCTIONS_SETTINGS,
    )
    def check_groundedness(item: dict) -> str:
        context_block = "\n\n---\n\n".join(item["contexts"])
        return f"Question: {item['question']}\n\nContext:\n{context_block}\n\nAnswer:\n{item['answer']}"

    @task
    def zip_groundedness_verdicts(items: list[dict], verdicts: list) -> list[dict]:
        def as_dict(v):
            return v if isinstance(v, dict) else v.model_dump()
        return [{"question": item["question"], "verdict": as_dict(v)} for item, v in zip(items, verdicts)]

    @task.branch(
        retries=4,
        retry_delay=timedelta(minutes=1),
        retry_exponential_backoff=True,
        max_retry_delay=timedelta(minutes=15),
    )
    def evaluate_quality_gates(manifest: dict, generated_answers: list[dict], groundedness_verdicts: list[dict]) -> str:
        from src.evaluate import run_all_gates, record_manifest_history
        from src.ingest import IngestManifest

        manifest_obj = IngestManifest(**manifest)
        results = run_all_gates(manifest_obj, SOURCE_DIR, generated_answers, groundedness_verdicts)

        for r in results:
            print(f"[quality-gate] {r.name}: {'PASS' if r.passed else 'FAIL'} -- {r.detail}")

        all_passed = all(r.passed for r in results)
        if all_passed:
            record_manifest_history(manifest_obj)
            return "promote_to_prod"

        pii_result = next(r for r in results if r.name == "pii_hard_block")
        if not pii_result.passed:
            return "security_incident_review"
        return "quality_gate_review"

    @task(outlets=[postmortem_index_prod])
    def promote_to_prod() -> str:
        from src.ingest import promote_staging_to_prod

        promote_staging_to_prod()
        return "promoted"

    security_incident_review = ApprovalOperator(
        task_id="security_incident_review",
        subject="URGENT: secret/credential leak detected in postmortem RAG staging index",
        body=(
            "The pii_hard_block quality gate found a hard-severity secret (e.g. an AWS "
            "access key) in staged content. Prod index was NOT updated.\n\n"
            "Action needed: rotate/redact the leaked credential and confirm before the "
            "source postmortem is re-ingested. See the evaluate_quality_gates task log "
            "for the exact finding(s)."
        ),
        response_timeout=timedelta(hours=4),
        notifiers=SlackWebhookNotifier(
            slack_webhook_conn_id="slack_default",
            text=(
                ":rotating_light: *Secret leak detected in postmortem RAG staging index*\n"
                "`pii_hard_block` quality gate failed -- prod index was NOT updated.\n"
                "A human needs to rotate/redact the credential and approve the "
                "`security_incident_review` task in Airflow before re-ingesting.\n"
                "DAG: `postmortem_rag_pipeline`"
            ),
        ),
    )

    quality_gate_review = ApprovalOperator(
        task_id="quality_gate_review",
        subject="Postmortem RAG staging promotion blocked by a quality gate",
        body=(
            "One or more non-security quality gates failed. Prod index was NOT updated. "
            "See the evaluate_quality_gates task log for per-check detail."
        ),
        response_timeout=timedelta(days=1),
    )

    @task(trigger_rule="none_failed_min_one_success")
    def block_promotion() -> None:
        raise ValueError(
            "One or more quality gates/guardrails failed. Prod index was NOT updated. "
            "See task logs above for per-check detail."
        )

    manifest = build_staging_index()
    staged = publish_staging_asset(manifest)

    eval_items = retrieve_eval_contexts(staged)
    eval_answers = generate_eval_answer.expand(item=eval_items)
    generated = zip_generated_answers(eval_items, eval_answers)

    groundedness_raw = check_groundedness.expand(item=generated)
    groundedness_verdicts = zip_groundedness_verdicts(generated, groundedness_raw)

    branch = evaluate_quality_gates(staged, generated, groundedness_verdicts)
    block = block_promotion()
    branch >> promote_to_prod()
    branch >> security_incident_review >> block
    branch >> quality_gate_review >> block

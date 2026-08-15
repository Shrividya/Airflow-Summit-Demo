from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow.providers.slack.notifications.slack_webhook import SlackWebhookNotifier
from airflow.providers.standard.operators.hitl import HITLOperator
from airflow.sdk import DAG, Asset, task
from include.guardrails import (
    CACHED_INSTRUCTIONS_SETTINGS,
    GROUNDEDNESS_GUARDRAIL_SYSTEM_PROMPT,
    GROUNDEDNESS_OUTPUT_TYPE,
    INFRA_TASK_RETRY_KWARGS,
    LLM_TASK_RETRY_KWARGS,
)

SOURCE_DIR = os.environ.get("PM_RAG_SOURCE_DIR", "/usr/local/airflow/data/postmortems")
EVAL_DATASET_PATH = os.environ.get("PM_RAG_EVAL_DATASET", "/usr/local/airflow/data/eval_dataset.jsonl")
LLM_CONN_ID = os.environ.get("PM_RAG_LLM_CONN_ID", "pydanticai_default")
GROUNDEDNESS_LLM_CONN_ID = os.environ.get("PM_RAG_GROUNDEDNESS_LLM_CONN_ID", "pydanticai_groundedness")

GENERATION_SYSTEM_PROMPT = (
    "Answer the question using ONLY the postmortem excerpts provided. "
    "If the excerpts don't contain the answer, say so."
)

postmortem_index_staging = Asset("postmortem_index_staging")
postmortem_index_prod = Asset("postmortem_index_prod")

QUALITY_GATE_ACK_OPTION = "Acknowledged - block promotion"
QUALITY_GATE_OVERRIDE_OPTION = "Override - promote anyway"

with DAG(
    dag_id="postmortem_rag_pipeline",
    description="Ingest incident postmortems into a quality-gated, guardrailed RAG index",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["rag", "quality-gates", "guardrails", "common-ai", "reference-architecture"],
    rerun_with_latest_version=False,
    # longer than quality_gate_review's 1-day response_timeout so a pending
    # human approval doesn't get killed out from under it.
    dagrun_timeout=timedelta(days=2),
):

    @task(**INFRA_TASK_RETRY_KWARGS)
    def plan_ingestion() -> dict:
        from include.ingest import plan_incremental_ingest
        return plan_incremental_ingest(SOURCE_DIR)


    @task
    def docs_to_embed(plan: dict) -> list[str]:
        return plan["to_embed"]


    @task(**LLM_TASK_RETRY_KWARGS)
    def embed_document(doc_id: str, ingest_run_id: str) -> dict:
        from include.ingest import embed_document as _embed_document
        return _embed_document(SOURCE_DIR, doc_id, ingest_run_id)



    @task(**INFRA_TASK_RETRY_KWARGS)
    def assemble_index(plan: dict, embedded_docs: list[dict]) -> dict:
        from include.ingest import assemble_staging_index
        manifest = assemble_staging_index(plan, embedded_docs)
        return manifest.__dict__


    @task(outlets=[postmortem_index_staging])
    def publish_staging_asset(manifest: dict) -> dict:
        return manifest


    @task
    def grow_golden_set() -> dict:
        from include.evaluate import add_frequent_questions_to_golden_set
        result = add_frequent_questions_to_golden_set(EVAL_DATASET_PATH)
        if result["added"]:
            print(f"[grow_golden_set] Added {len(result['added'])} frequently asked question(s): {result['added']}")
        else:
            print(f"[grow_golden_set] No new questions met the frequency threshold ({result['candidates_seen']} candidates seen).")
        return result


    @task
    def retrieve_eval_contexts(_manifest: dict) -> list[dict]:
        from include.ingest import STAGING_COLLECTION, retrieve
        with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        items = []
        for row in rows:
            hits = retrieve(row["question"], collection_name=STAGING_COLLECTION, k=4)
            items.append({
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "contexts": [h["text"] for h in hits],
            })
        return items

    @task.llm(
        llm_conn_id=LLM_CONN_ID,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        agent_params=CACHED_INSTRUCTIONS_SETTINGS,
        **LLM_TASK_RETRY_KWARGS,
    )
    def generate_eval_answer(item: dict) -> str:
        context_block = "\n\n---\n\n".join(item["contexts"])
        return f"Excerpts:\n{context_block}\n\nQuestion: {item['question']}\nAnswer:"

    @task
    def zip_generated_answers(items: list[dict], answers: list[str]) -> list[dict]:
        return [{**item, "answer": answer} for item, answer in zip(items, answers, strict=True)]

    @task.llm(
        llm_conn_id=GROUNDEDNESS_LLM_CONN_ID,
        system_prompt=GROUNDEDNESS_GUARDRAIL_SYSTEM_PROMPT,
        output_type=GROUNDEDNESS_OUTPUT_TYPE,
        serialize_output=True,
        agent_params=CACHED_INSTRUCTIONS_SETTINGS,
        **LLM_TASK_RETRY_KWARGS,
    )
    def check_groundedness(item: dict) -> str:
        context_block = "\n\n---\n\n".join(item["contexts"])
        return f"Question: {item['question']}\n\nContext:\n{context_block}\n\nAnswer:\n{item['answer']}"

    @task
    def zip_groundedness_verdicts(items: list[dict], verdicts: list) -> list[dict]:
        def as_dict(v):
            return v if isinstance(v, dict) else v.model_dump()
        return [{"question": item["question"], "verdict": as_dict(v)} for item, v in zip(items, verdicts, strict=True)]

    def _alert_promotion_stalled(streak: int, results) -> None:
        failing = "; ".join(f"{r.name}: {r.detail}" for r in results if not r.passed)
        SlackWebhookNotifier(
            slack_webhook_conn_id="slack_default",
            text=(
                f":warning: *Postmortem RAG staging hasn't promoted in {streak} consecutive runs*\n"
                f"Failing gate(s) this run: {failing}\n"
                "Prod index is increasingly stale. See the evaluate_quality_gates task log for detail.\n"
                "DAG: `postmortem_rag_pipeline`"
            ),
        ).notify({})

    @task.branch(**INFRA_TASK_RETRY_KWARGS)
    def evaluate_quality_gates(manifest: dict, generated_answers: list[dict], groundedness_verdicts: list[dict]) -> str:
        from include.evaluate import (
            BLOCKED_RUN_ALERT_THRESHOLD,
            record_manifest_history,
            record_promotion_outcome,
            run_all_gates,
        )
        from include.ingest import IngestManifest

        manifest_obj = IngestManifest(**manifest)
        results = run_all_gates(manifest_obj, SOURCE_DIR, generated_answers, groundedness_verdicts)

        for r in results:
            print(f"[quality-gate] {r.name}: {'PASS' if r.passed else 'FAIL'} -- {r.detail}")

        all_passed = all(r.passed for r in results)
        if all_passed:
            record_manifest_history(manifest_obj)
            record_promotion_outcome(promoted=True)
            return "promote_to_prod"

        streak = record_promotion_outcome(promoted=False)
        print(f"[quality-gate] staging has not promoted for {streak} consecutive run(s).")
        if streak >= BLOCKED_RUN_ALERT_THRESHOLD:
            _alert_promotion_stalled(streak, results)

        pii_result = next(r for r in results if r.name == "pii_hard_block")
        if not pii_result.passed:
            return "security_incident_review"
        return "quality_gate_review"

    @task(outlets=[postmortem_index_prod], trigger_rule="none_failed_min_one_success")
    def promote_to_prod() -> str:
        from include.ingest import promote_staging_to_prod

        promote_staging_to_prod()
        return "promoted"

    security_incident_review = HITLOperator(
        task_id="security_incident_review",
        subject="URGENT: secret/credential leak detected in postmortem RAG staging index",
        body=(
            "The pii_hard_block quality gate found a hard-severity secret (e.g. an AWS "
            "access key) in staged content. Prod index was NOT updated.\n\n"
            "Action needed: rotate/redact the leaked credential and confirm before the "
            "source postmortem is re-ingested. See the evaluate_quality_gates task log "
            "for the exact finding(s)."
        ),
        options=["Acknowledged"],
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

    quality_gate_review = HITLOperator(
        task_id="quality_gate_review",
        subject="Postmortem RAG staging promotion blocked by a quality gate",
        body=(
            "One or more non-security quality gates failed. Prod index was NOT updated. "
            "See the evaluate_quality_gates task log for per-check detail.\n\n"
            f'Choose "{QUALITY_GATE_OVERRIDE_OPTION}" only if you have confirmed the '
            "failing gate(s) are a false positive for this run -- this promotes staging "
            "to prod as-is, skipping the gate(s) that failed."
        ),
        options=[QUALITY_GATE_ACK_OPTION, QUALITY_GATE_OVERRIDE_OPTION],
        response_timeout=timedelta(days=1),
    )

    @task.branch
    def route_quality_gate_review(response: dict) -> str:
        if QUALITY_GATE_OVERRIDE_OPTION in response["chosen_options"]:
            return "promote_to_prod"
        return "block_promotion"

    @task(trigger_rule="none_failed_min_one_success")
    def block_promotion() -> None:
        raise ValueError(
            "One or more quality gates/guardrails failed. Prod index was NOT updated. "
            "See task logs above for per-check detail."
        )

    plan = plan_ingestion()
    to_embed = docs_to_embed(plan)
    embedded = embed_document.partial(ingest_run_id=plan["run_id"]).expand(doc_id=to_embed)
    manifest = assemble_index(plan, embedded)
    staged = publish_staging_asset(manifest)
    golden_set_grown = grow_golden_set()

    eval_items = retrieve_eval_contexts(staged)
    golden_set_grown >> eval_items
    eval_answers = generate_eval_answer.expand(item=eval_items)
    generated = zip_generated_answers(eval_items, eval_answers)

    groundedness_raw = check_groundedness.expand(item=generated)
    groundedness_verdicts = zip_groundedness_verdicts(generated, groundedness_raw)

    branch = evaluate_quality_gates(staged, generated, groundedness_verdicts)
    promote = promote_to_prod()
    block = block_promotion()
    branch >> promote
    branch >> security_incident_review >> block

    quality_gate_route = route_quality_gate_review(quality_gate_review.output)
    branch >> quality_gate_review >> quality_gate_route
    quality_gate_route >> promote
    quality_gate_route >> block

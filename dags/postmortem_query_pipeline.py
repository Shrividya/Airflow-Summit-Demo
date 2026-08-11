"""Answer a question against the PRODUCTION postmortem index. check_input_safety
blocks prompt-injection/off-topic questions before retrieval; check_groundedness
blocks an answer whose claims aren't supported by the retrieved context.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

from airflow.sdk import DAG, Param, get_current_context, task

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.guardrails import (
    CACHED_INSTRUCTIONS_SETTINGS,
    GROUNDEDNESS_GUARDRAIL_SYSTEM_PROMPT,
    GROUNDEDNESS_OUTPUT_TYPE,
    INPUT_GUARDRAIL_SYSTEM_PROMPT,
    INPUT_SAFETY_OUTPUT_TYPE,
    LLM_TASK_RETRY_KWARGS,
    GroundednessVerdict,
    InputSafetyVerdict,
)

LLM_CONN_ID = os.environ.get("PM_RAG_LLM_CONN_ID", "pydanticai_default")

GENERATION_SYSTEM_PROMPT = (
    "Answer the question using ONLY the postmortem excerpts below. "
    "If the excerpts don't contain the answer, say so."
)


def _record_result(status: str, text: str, sources: list[dict] | None = None) -> None:
    run_id = get_current_context()["dag_run"].run_id
    db_path = os.path.join(os.environ.get("AIRFLOW_HOME", os.getcwd()), "include", "query_results.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS query_results ("
            "run_id TEXT PRIMARY KEY, status TEXT NOT NULL, text TEXT NOT NULL, "
            "sources_json TEXT, ts TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO query_results (run_id, status, text, sources_json, ts) VALUES (?,?,?,?,?)",
            (run_id, status, text, json.dumps(sources) if sources else None, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


with DAG(
    dag_id="postmortem_query_pipeline",
    description="Answer a question against the PROD postmortem index, with input-safety and groundedness guardrails",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["rag", "guardrails", "common-ai", "reference-architecture"],
    params={"question": Param("", type="string", description="The question to ask the postmortem RAG assistant.")},
):

    @task.llm(
        llm_conn_id=LLM_CONN_ID,
        system_prompt=INPUT_GUARDRAIL_SYSTEM_PROMPT,
        output_type=INPUT_SAFETY_OUTPUT_TYPE,
        serialize_output=True,
        agent_params=CACHED_INSTRUCTIONS_SETTINGS,
        **LLM_TASK_RETRY_KWARGS,
    )
    def check_input_safety(question: str) -> str:
        return question

    @task.branch
    def route_on_input_safety(verdict: InputSafetyVerdict) -> str:
        v = verdict if isinstance(verdict, dict) else verdict.model_dump()
        return "retrieve_context" if v["safe"] else "refuse"

    @task
    def refuse(verdict: InputSafetyVerdict) -> None:
        v = verdict if isinstance(verdict, dict) else verdict.model_dump()
        _record_result("refused", f"I can't answer that: {v['reason']}")
        raise ValueError(f"Input-safety guardrail refused this question: {v['reason']}")

    @task
    def retrieve_context(question: str) -> dict:
        from src.ingest import retrieve, PROD_COLLECTION

        hits = retrieve(question, collection_name=PROD_COLLECTION, k=4)
        if not hits:
            msg = "No production index found yet -- run postmortem_rag_pipeline at least once."
            _record_result("error", msg)
            raise ValueError(msg)
        return {
            "question": question,
            "contexts": [h["text"] for h in hits],
            "sources": [
                {"doc_id": h["metadata"]["doc_id"], "chunk_index": h["metadata"]["chunk_index"], "distance": h["distance"]}
                for h in hits
            ],
        }

    @task.llm(
        llm_conn_id=LLM_CONN_ID,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        agent_params=CACHED_INSTRUCTIONS_SETTINGS,
        **LLM_TASK_RETRY_KWARGS,
    )
    def generate_answer(retrieved: dict) -> str:
        context_block = "\n\n---\n\n".join(retrieved["contexts"])
        return f"Excerpts:\n{context_block}\n\nQuestion: {retrieved['question']}\nAnswer:"

    @task.llm(
        llm_conn_id=LLM_CONN_ID,
        system_prompt=GROUNDEDNESS_GUARDRAIL_SYSTEM_PROMPT,
        output_type=GROUNDEDNESS_OUTPUT_TYPE,
        serialize_output=True,
        agent_params=CACHED_INSTRUCTIONS_SETTINGS,
        **LLM_TASK_RETRY_KWARGS,
    )
    def check_groundedness(retrieved: dict, answer: str) -> str:
        context_block = "\n\n---\n\n".join(retrieved["contexts"])
        return f"Question: {retrieved['question']}\n\nContext:\n{context_block}\n\nAnswer:\n{answer}"

    @task.branch
    def route_on_groundedness(verdict: GroundednessVerdict) -> str:
        v = verdict if isinstance(verdict, dict) else verdict.model_dump()
        return "deliver_answer" if v["grounded"] else "block_answer"

    @task
    def deliver_answer(retrieved: dict, answer: str) -> str:
        print("\n=== ANSWER ===")
        print(answer)
        print("\n=== SOURCES ===")
        for s in retrieved["sources"]:
            print(f"- {s['doc_id']} (chunk {s['chunk_index']}, distance={s['distance']:.3f})")
        _record_result("answered", answer, sources=retrieved["sources"])
        return answer

    @task
    def block_answer(verdict: GroundednessVerdict) -> None:
        v = verdict if isinstance(verdict, dict) else verdict.model_dump()
        _record_result(
            "blocked",
            f"I found an answer but couldn't verify it's grounded in the postmortems: {v['reason']}",
        )
        raise ValueError(
            "Groundedness guardrail blocked this answer -- unsupported claims: "
            f"{v['unsupported_claims']} ({v['reason']})"
        )

    input_verdict = check_input_safety(question="{{ params.question }}")
    safety_branch = route_on_input_safety(input_verdict)

    retrieved = retrieve_context(question="{{ params.question }}")
    answer = generate_answer(retrieved)
    groundedness_verdict = check_groundedness(retrieved, answer)
    groundedness_branch = route_on_groundedness(groundedness_verdict)

    safety_branch >> refuse(input_verdict)
    safety_branch >> retrieved
    groundedness_branch >> deliver_answer(retrieved, answer)
    groundedness_branch >> block_answer(groundedness_verdict)

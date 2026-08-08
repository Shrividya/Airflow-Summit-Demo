"""Local UI: ask the postmortem RAG assistant a question without touching the
Airflow UI or the CLI (`src/query.py`).

Runs as a plain local process (`streamlit run streamlit_app.py`), outside the
Astro/Airflow containers. It triggers postmortem_query_pipeline over Airflow's
REST API, shows the pipeline's real task states as it runs (input-safety ->
retrieval -> generation -> groundedness), then reads the outcome from
include/query_results.db -- written by `_record_result` in
dags/postmortem_query_pipeline.py -- keyed by the dag_run_id this script
picks. That file is readable directly because Astro dev bind-mounts the
project directory into the containers, so the same file a task writes from
inside a container is right here on disk.

    pip install streamlit requests
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid

import requests
import streamlit as st

AIRFLOW_BASE_URL = os.environ.get("AIRFLOW_BASE_URL", "http://localhost:8080")
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "admin")
DAG_ID = "postmortem_query_pipeline"
RESULTS_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include", "query_results.db")
POLL_INTERVAL_SECONDS = 1
POLL_TIMEOUT_SECONDS = 90

# Ordered so the status view fills in top-to-bottom as the DAG actually runs;
# only the tasks on the "safe, grounded" path get a friendly label -- refuse/
# block_answer show up implicitly once query_results.db has the final verdict.
PIPELINE_STAGES = [
    ("check_input_safety", "Checking the question isn't off-topic or a prompt injection"),
    ("retrieve_context", "Retrieving relevant postmortem excerpts"),
    ("generate_answer", "Generating an answer from those excerpts"),
    ("check_groundedness", "Verifying the answer is grounded in the excerpts"),
]

STATUS_STYLE = {
    "answered": ("success", "✅"),
    "refused": ("warning", "🚫"),
    "blocked": ("warning", "⚠️"),
}

EXAMPLE_QUESTIONS = [
    "What caused the checkout latency spike?",
    "What caused the support ticket leak incident?",
    "Why did customers get charged twice?",
]

st.set_page_config(page_title="Postmortem RAG Assistant", page_icon="🔎", layout="centered")


def _get_token() -> str:
    resp = requests.post(
        f"{AIRFLOW_BASE_URL}/auth/token",
        json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _trigger_dag_run(question: str, token: str) -> str:
    run_id = f"streamlit__{uuid.uuid4().hex}"
    resp = requests.post(
        f"{AIRFLOW_BASE_URL}/api/v2/dags/{DAG_ID}/dagRuns",
        headers={"Authorization": f"Bearer {token}"},
        json={"dag_run_id": run_id, "logical_date": None, "conf": {"question": question}},
        timeout=10,
    )
    resp.raise_for_status()
    return run_id


def _task_states(run_id: str, token: str) -> dict[str, str]:
    try:
        resp = requests.get(
            f"{AIRFLOW_BASE_URL}/api/v2/dags/{DAG_ID}/dagRuns/{run_id}/taskInstances",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return {}
    return {ti["task_id"]: ti["state"] for ti in resp.json().get("task_instances", [])}


def _poll_db(run_id: str) -> tuple[str, str, list[dict]] | None:
    if not os.path.exists(RESULTS_DB_PATH):
        return None
    conn = sqlite3.connect(RESULTS_DB_PATH, timeout=30)
    try:
        row = conn.execute(
            "SELECT status, text, sources_json FROM query_results WHERE run_id = ?", (run_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        row = None  # table not created yet
    finally:
        conn.close()
    if not row:
        return None
    status, text, sources_json = row
    return status, text, (json.loads(sources_json) if sources_json else [])


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"- **{s['doc_id']}** (chunk {s['chunk_index']}, distance={s['distance']:.3f})")


def _render_result(status: str, text: str, sources: list[dict]) -> None:
    kind, icon = STATUS_STYLE.get(status, ("info", "ℹ️"))
    getattr(st, kind)(f"{icon} {text}")
    _render_sources(sources)


def _run_question(question: str, placeholder) -> dict:
    """Trigger the DAG, live-update `placeholder` with pipeline progress, and
    return {"status", "text", "sources"} once query_results.db has a row (or
    a synthetic "error"/"timeout" status).
    """
    try:
        token = _get_token()
        run_id = _trigger_dag_run(question, token)
    except requests.RequestException as e:
        return {"status": "error", "text": f"Couldn't reach Airflow at {AIRFLOW_BASE_URL}: {e}", "sources": []}

    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    with placeholder.status("Running the guarded RAG pipeline...", expanded=True) as status_box:
        st.caption(f"Run: `{run_id}`")
        while time.monotonic() < deadline:
            db_row = _poll_db(run_id)
            if db_row is not None:
                status, text, sources = db_row
                label = "Answer ready" if status == "answered" else "Blocked by a guardrail"
                status_box.update(label=label, state="complete" if status == "answered" else "error")
                return {"status": status, "text": text, "sources": sources}

            states = _task_states(run_id, token)
            for task_id, label in PIPELINE_STAGES:
                state = states.get(task_id)
                if state == "success":
                    st.write(f"✅ {label}")
                elif state in ("running", "queued", "scheduled"):
                    st.write(f"⏳ {label}")
                elif state == "failed":
                    st.write(f"❌ {label}")
            time.sleep(POLL_INTERVAL_SECONDS)

        status_box.update(label="Timed out waiting for a result", state="error")
        return {
            "status": "error",
            "text": f"No result after {POLL_TIMEOUT_SECONDS}s -- check the Grid UI for run `{run_id}`.",
            "sources": [],
        }


st.title("🔎 Postmortem RAG Assistant")
st.caption(f"Asks {DAG_ID} on {AIRFLOW_BASE_URL} -- every answer passes through the same guardrails as the CLI.")

with st.sidebar:
    st.subheader("Try one of these")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q
    st.divider()
    st.caption(
        "Guardrails: input-safety blocks off-topic/prompt-injection questions "
        "before retrieval; groundedness blocks an answer whose claims aren't "
        "backed by the retrieved postmortems."
    )

if "history" not in st.session_state:
    st.session_state["history"] = []  # list of {"question", "status", "text", "sources"}

for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        _render_result(turn["status"], turn["text"], turn["sources"])

typed_question = st.chat_input("Ask about a past incident...")
question = typed_question or st.session_state.pop("pending_question", None)

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        result = _run_question(question, st.empty())
        _render_result(result["status"], result["text"], result["sources"])
    st.session_state["history"].append({"question": question, **result})

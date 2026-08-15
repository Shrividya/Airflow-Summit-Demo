from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from include.ingest import CHROMA_PATH, IngestManifest

GENERATION_MODEL = os.environ.get("PM_RAG_GENERATION_MODEL", "meta-llama/Llama-3.3-70B-Instruct:novita")
HF_BASE_URL = os.environ.get("HF_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.environ.get("HF_TOKEN")
RAGAS_CACHE_DIR = os.environ.get("PM_RAG_RAGAS_CACHE_DIR", "/tmp/postmortem-rag-ragas-cache")

# same computation as dags/postmortem_query_pipeline.py's _record_result
QUERY_RESULTS_DB_PATH = os.environ.get(
    "PM_RAG_QUERY_RESULTS_DB",
    os.path.join(os.environ.get("AIRFLOW_HOME", os.getcwd()), "include", "query_results.db"),
)
GOLDEN_SET_MIN_ASKED_COUNT = int(os.environ.get("PM_RAG_GOLDEN_SET_MIN_ASKED_COUNT", "3"))
BLOCKED_RUN_ALERT_THRESHOLD = int(os.environ.get("PM_RAG_BLOCKED_RUN_ALERT_THRESHOLD", "3"))

# allow chunk count/size to move by this much run-over-run before blocking
MAX_CHUNK_COUNT_DRIFT_PCT = 0.25
MAX_AVG_CHUNK_CHARS_DRIFT_PCT = 0.30

RAGAS_FLOORS = {
    "faithfulness": 0.85,
    "context_precision": 0.75,
    "context_recall": 0.75,
}

MANIFEST_HISTORY_PATH = Path(CHROMA_PATH) / "manifest_history.json"
PROMOTION_STREAK_PATH = Path(CHROMA_PATH) / "promotion_streak.json"


@dataclass
class QualityGateResult:
    name: str
    passed: bool
    detail: str
    metrics: dict | None = None


def _load_previous_manifest() -> dict | None:
    if not MANIFEST_HISTORY_PATH.exists():
        return None
    history = json.loads(MANIFEST_HISTORY_PATH.read_text())
    return history[-1] if history else None


def record_manifest_history(manifest: IngestManifest) -> None:
    history = []
    if MANIFEST_HISTORY_PATH.exists():
        history = json.loads(MANIFEST_HISTORY_PATH.read_text())
    history.append(manifest.__dict__)
    MANIFEST_HISTORY_PATH.write_text(json.dumps(history[-20:], indent=2))  # keep last 20 runs


def record_promotion_outcome(promoted: bool) -> int:
    """Track consecutive blocked (not-promoted) runs so the DAG can alert once
    staging has failed to promote BLOCKED_RUN_ALERT_THRESHOLD times in a row.
    Resets to 0 on any promoted run. Returns the streak count *after* this run."""
    streak = 0
    if PROMOTION_STREAK_PATH.exists():
        streak = json.loads(PROMOTION_STREAK_PATH.read_text()).get("consecutive_blocked", 0)
    streak = 0 if promoted else streak + 1
    PROMOTION_STREAK_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMOTION_STREAK_PATH.write_text(json.dumps({"consecutive_blocked": streak}))
    return streak


def _normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _load_golden_questions(eval_dataset_path: str) -> set[str]:
    path = Path(eval_dataset_path)
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {_normalize_question(r["question"]) for r in rows}


def add_frequent_questions_to_golden_set(
    eval_dataset_path: str,
    min_asked_count: int = GOLDEN_SET_MIN_ASKED_COUNT,
    query_results_db_path: str = QUERY_RESULTS_DB_PATH,
) -> dict:
    """Grow the golden eval set from real production usage: any question that
    was asked at least `min_asked_count` times and successfully answered
    (i.e. passed both the input-safety and groundedness guardrails in
    dags/postmortem_query_pipeline.py -- status "answered") gets appended,
    using its most recently delivered answer as ground_truth. Questions
    already present in the golden set (by normalized text) are skipped."""
    if not Path(query_results_db_path).exists():
        return {"added": [], "candidates_seen": 0}

    conn = sqlite3.connect(query_results_db_path, timeout=30)
    try:
        rows = conn.execute(
            "SELECT question, text, ts FROM query_results "
            "WHERE status = 'answered' AND question IS NOT NULL ORDER BY ts"
        ).fetchall()
    except sqlite3.OperationalError:
        return {"added": [], "candidates_seen": 0}  # table doesn't exist yet
    finally:
        conn.close()

    by_question: dict[str, dict] = {}
    for question, answer, _ts in rows:
        key = _normalize_question(question)
        entry = by_question.setdefault(key, {"count": 0})
        entry["count"] += 1
        entry["question"] = question.strip()  # most recent casing/phrasing wins
        entry["answer"] = answer  # most recent grounded answer wins

    existing = _load_golden_questions(eval_dataset_path)
    to_add = [
        e for key, e in by_question.items()
        if e["count"] >= min_asked_count and key not in existing
    ]

    if to_add:
        with open(eval_dataset_path, "a") as f:
            for e in to_add:
                f.write(json.dumps({"question": e["question"], "ground_truth": e["answer"]}) + "\n")

    return {"added": [e["question"] for e in to_add], "candidates_seen": len(by_question)}


def check_chunking_regression(manifest: IngestManifest) -> QualityGateResult:
    prev = _load_previous_manifest()
    if prev is None:
        return QualityGateResult("chunking_regression", True, "No previous run to compare against; baseline accepted.")

    count_drift = abs(manifest.chunk_count - prev["chunk_count"]) / max(prev["chunk_count"], 1)
    size_drift = abs(manifest.avg_chunk_chars - prev["avg_chunk_chars"]) / max(prev["avg_chunk_chars"], 1)

    passed = count_drift <= MAX_CHUNK_COUNT_DRIFT_PCT and size_drift <= MAX_AVG_CHUNK_CHARS_DRIFT_PCT
    detail = (
        f"chunk_count {prev['chunk_count']} -> {manifest.chunk_count} ({count_drift:.0%} drift), "
        f"avg_chunk_chars {prev['avg_chunk_chars']:.0f} -> {manifest.avg_chunk_chars:.0f} ({size_drift:.0%} drift)"
    )
    return QualityGateResult("chunking_regression", passed, detail, {
        "count_drift_pct": count_drift, "size_drift_pct": size_drift,
    })


def check_embedding_model_drift(manifest: IngestManifest) -> QualityGateResult:
    prev = _load_previous_manifest()
    if prev is None:
        return QualityGateResult("embedding_model_drift", True, "No previous run to compare against; baseline accepted.")

    passed = manifest.embedding_model == prev["embedding_model"]
    detail = (
        f"embedding_model unchanged: {manifest.embedding_model}"
        if passed else
        f"embedding_model changed from {prev['embedding_model']} to {manifest.embedding_model} -- requires a full re-index"
    )
    return QualityGateResult("embedding_model_drift", passed, detail)


def check_partial_reindex(manifest: IngestManifest, expected_source_dir: str) -> QualityGateResult:
    from include.ingest import load_source_documents

    source_docs = load_source_documents(expected_source_dir)
    expected_ids = set(source_docs.keys())
    indexed_ids = set(manifest.source_doc_hashes.keys())

    missing = expected_ids - indexed_ids
    passed = len(missing) == 0
    detail = (
        f"All {len(expected_ids)} source documents present in staging index."
        if passed else
        f"Staging index is missing {len(missing)} of {len(expected_ids)} source documents: {sorted(missing)}"
    )
    return QualityGateResult("partial_reindex", passed, detail)


def check_pii_hard_block(manifest: IngestManifest) -> QualityGateResult:
    if not manifest.has_hard_block:
        return QualityGateResult("pii_hard_block", True, "No hard-severity PII/secret findings in this run.")

    hard_docs = {
        doc_id: [f for f in findings if f["severity"] == "hard"]
        for doc_id, findings in manifest.pii_findings.items()
        if any(f["severity"] == "hard" for f in findings)
    }
    return QualityGateResult(
        "pii_hard_block", False,
        f"Hard-severity PII/secret findings in: {hard_docs} -- requires human review before re-ingesting.",
        {"hard_finding_doc_count": len(hard_docs)},
    )


def check_answer_guardrails(verdicts: list[dict]) -> QualityGateResult:
    ungrounded = [v for v in verdicts if not v["verdict"]["grounded"]]
    passed = len(ungrounded) == 0
    detail = (
        f"All {len(verdicts)} golden-set answers judged grounded."
        if passed else
        f"{len(ungrounded)} of {len(verdicts)} golden-set answers judged ungrounded: "
        + "; ".join(f"{u['question']!r}: {u['verdict']['reason']}" for u in ungrounded)
    )
    return QualityGateResult("answer_guardrails", passed, detail, {"ungrounded_count": len(ungrounded)})


def check_retrieval_quality(rows: list[dict]) -> QualityGateResult:
    """`rows` are dicts with question/contexts/ground_truth/answer, one per golden question."""
    import instructor
    from datasets import Dataset
    from openai import OpenAI
    from ragas import evaluate
    from ragas.cache import DiskCacheBackend
    from ragas.llms.base import InstructorLLM
    from ragas.metrics import context_precision, context_recall, faithfulness

    questions = [r["question"] for r in rows]
    ground_truths = [r["ground_truth"] for r in rows]
    answers = [r["answer"] for r in rows]
    contexts_list = [r["contexts"] for r in rows]

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # novita rejects response_format: json_object/json_schema (same constraint
    # as include/guardrails.py) but supports tool-calling, so use Mode.TOOLS
    patched_client = instructor.from_openai(
        OpenAI(base_url=HF_BASE_URL, api_key=HF_TOKEN), mode=instructor.Mode.TOOLS
    )
    ragas_cache = DiskCacheBackend(cache_dir=RAGAS_CACHE_DIR)
    ragas_llm = InstructorLLM(client=patched_client, model=GENERATION_MODEL, provider="openai", cache=ragas_cache)
    # this model rejects ragas's default temperature=0.01/top_p
    if hasattr(ragas_llm, "model_args"):
        ragas_llm.model_args.pop("temperature", None)
        ragas_llm.model_args.pop("top_p", None)
    result = evaluate(ds, metrics=[faithfulness, context_precision, context_recall], llm=ragas_llm)

    scores = {}
    for name in RAGAS_FLOORS:
        try:
            value = result[name]
        except (KeyError, TypeError):
            continue
        if isinstance(value, (list, tuple)):
            value = sum(value) / len(value) if value else 0.0
        scores[name] = float(value)

    failures = [
        f"{metric}={scores.get(metric, 0):.2f} < floor {floor}"
        for metric, floor in RAGAS_FLOORS.items()
        if scores.get(metric, 0) < floor
    ]
    passed = len(failures) == 0
    scores_summary = "; ".join(f"{metric}={scores.get(metric, 0):.2f} (floor {floor})" for metric, floor in RAGAS_FLOORS.items())
    detail = (
        f"All RAGAS metrics at or above floor -- {scores_summary}." if passed
        else "Below floor: " + "; ".join(failures)
    )
    return QualityGateResult("retrieval_quality_decay", passed, detail, scores)


def run_all_gates(
    manifest: IngestManifest,
    source_dir: str,
    generated_answers: list[dict],
    groundedness_verdicts: list[dict],
) -> list[QualityGateResult]:
    """`generated_answers` and `groundedness_verdicts` come from the DAG's mapped tasks."""
    return [
        check_chunking_regression(manifest),
        check_embedding_model_drift(manifest),
        check_partial_reindex(manifest, source_dir),
        check_pii_hard_block(manifest),
        check_retrieval_quality(generated_answers),
        check_answer_guardrails(groundedness_verdicts),
    ]

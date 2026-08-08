"""Quality gates for the postmortem RAG pipeline, run against the STAGING
index before promotion: structural checks (chunking/embedding drift,
partial re-index) diffed against the previous IngestManifest, RAGAS
evaluation scored against a floor, and guardrails that fail outright rather
than averaging away. Generation/groundedness verdicts come from the DAG's
mapped @task.llm tasks; this module just scores/branches.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.ingest import IngestManifest, CHROMA_PATH

GENERATION_MODEL = os.environ.get("PM_RAG_GENERATION_MODEL", "meta-llama/Llama-3.3-70B-Instruct:novita")
HF_BASE_URL = os.environ.get("HF_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.environ.get("HF_TOKEN")
RAGAS_CACHE_DIR = os.environ.get("PM_RAG_RAGAS_CACHE_DIR", "/tmp/postmortem-rag-ragas-cache")

# allow chunk count/size to move by this much run-over-run before blocking
MAX_CHUNK_COUNT_DRIFT_PCT = 0.25
MAX_AVG_CHUNK_CHARS_DRIFT_PCT = 0.30

RAGAS_FLOORS = {
    "faithfulness": 0.85,
    "context_precision": 0.75,
    "context_recall": 0.75,
}

MANIFEST_HISTORY_PATH = Path(CHROMA_PATH) / "manifest_history.json"


@dataclass
class QualityGateResult:
    name: str
    passed: bool
    detail: str
    metrics: Optional[dict] = None


def _load_previous_manifest() -> Optional[dict]:
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
        f"embedding_model changed from {prev['embedding_model']} to {manifest.embedding_model} "
        f"-- requires an explicit full re-index and reviewed config change, not an inline promotion"
    )
    return QualityGateResult("embedding_model_drift", passed, detail)


def check_partial_reindex(manifest: IngestManifest, expected_source_dir: str) -> QualityGateResult:
    from src.ingest import load_source_documents

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
    """`rows` are dicts with question/contexts/ground_truth/answer, one per
    golden question -- see generate_eval_answer in
    dags/postmortem_rag_pipeline.py."""
    import instructor
    from datasets import Dataset
    from ragas import evaluate
    from ragas.cache import DiskCacheBackend
    from ragas.metrics import faithfulness, context_precision, context_recall
    from ragas.llms.base import InstructorLLM
    from openai import OpenAI

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

    # novita (the HF router provider serving Llama-3.3-70B-Instruct here)
    # supports neither response_format: json_object nor json_schema
    # (supports_structured_output=False on the router's model listing) --
    # same constraint as INPUT_SAFETY_OUTPUT_TYPE/GROUNDEDNESS_OUTPUT_TYPE
    # in src/guardrails.py. Mode.TOOLS uses function-calling instead, which
    # novita does support (supports_tools=True).
    patched_client = instructor.from_openai(
        OpenAI(base_url=HF_BASE_URL, api_key=HF_TOKEN), mode=instructor.Mode.TOOLS
    )
    # cache judge responses on disk so reruns over unchanged staging content
    # don't repay the full RAGAS judging cost
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
    """`generated_answers` and `groundedness_verdicts` come from the DAG's
    mapped tasks (generate_eval_answer / check_groundedness), one entry
    per golden question."""
    return [
        check_chunking_regression(manifest),
        check_embedding_model_drift(manifest),
        check_partial_reindex(manifest, source_dir),
        check_pii_hard_block(manifest),
        check_retrieval_quality(generated_answers),
        check_answer_guardrails(groundedness_verdicts),
    ]

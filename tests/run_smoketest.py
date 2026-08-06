"""
Offline smoke test for the postmortem-rag project.

Uses hand-written fake chromadb / openai / datasets / ragas / airflow.sdk
modules (see tests/fakes/) so src/ingest.py and src/evaluate.py
run end-to-end without network access or real dependencies. Not a
substitute for running the real pipeline -- it only proves the Python
logic itself (chunking, manifest diffing, gate branching, DAG wiring) is
free of bugs.
"""
import json
import os
import sys
import shutil
import tempfile

FAKES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakes")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, FAKES_DIR)      # fake chromadb/openai/datasets/ragas/airflow
sys.path.insert(0, PROJECT_DIR)    # real src/

CHROMA_TMP = tempfile.mkdtemp(prefix="pm-rag-smoketest-")
os.environ["PM_RAG_CHROMA_PATH"] = CHROMA_TMP
os.environ["PM_RAG_EMBEDDING_MODEL"] = "nomic-embed-text"
os.environ["PM_RAG_GENERATION_MODEL"] = "llama3.1"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name} -- {detail}")


print("=== 1. build_staging_index ===")
from src.ingest import build_staging_index, STAGING_COLLECTION, PROD_COLLECTION, get_chroma_client, retrieve, promote_staging_to_prod

source_dir = os.path.join(PROJECT_DIR, "data", "postmortems")
manifest1 = build_staging_index(source_dir, run_id="run-1")

check("5 source docs loaded", manifest1.source_doc_count == 5, f"got {manifest1.source_doc_count}")
check("chunks were produced", manifest1.chunk_count > 0, f"got {manifest1.chunk_count}")
check("avg chunk size is reasonable (<=900 chars)", manifest1.avg_chunk_chars <= 900, f"got {manifest1.avg_chunk_chars}")
check("embedding model recorded", manifest1.embedding_model == "nomic-embed-text")
check("all 5 doc hashes recorded", len(manifest1.source_doc_hashes) == 5, f"got {list(manifest1.source_doc_hashes)}")

print("\n=== 1b. ingest-time PII/secret guardrail ===")
check(
    "INC-2350's email/token were found and flagged",
    "INC-2350-support-ticket-leak" in manifest1.pii_findings,
    manifest1.pii_findings,
)
check(
    "INC-2350's hard-severity finding (AWS key) set has_hard_block",
    manifest1.has_hard_block is True,
)

client = get_chroma_client()
staging_names = [c.name for c in client.list_collections()]
check("staging collection exists after build", STAGING_COLLECTION in staging_names, f"got {staging_names}")

staged_docs = client.get_collection(STAGING_COLLECTION).get()["documents"]
check(
    "raw AWS key never reaches the staging index (redacted before embedding)",
    not any("AKIAIOSFODNN7EXAMPLE" in d for d in staged_docs),
)
check(
    "raw customer email never reaches the staging index (redacted before embedding)",
    not any("jane.doe@example-customer.com" in d for d in staged_docs),
)
check(
    "redaction placeholder is present where the secret used to be",
    any("REDACTED_AWS_KEY" in d for d in staged_docs),
)

print("\n=== 2. quality gates against a fresh baseline (no prior history) ===")
from src.evaluate import (
    check_chunking_regression, check_embedding_model_drift, check_partial_reindex,
    check_pii_hard_block, check_answer_guardrails, record_manifest_history, MANIFEST_HISTORY_PATH,
)

r1 = check_chunking_regression(manifest1)
r2 = check_embedding_model_drift(manifest1)
r3 = check_partial_reindex(manifest1, source_dir)
check("chunking regression check passes with no history", r1.passed, r1.detail)
check("embedding drift check passes with no history", r2.passed, r2.detail)
check("partial reindex check passes (all 5 docs present)", r3.passed, r3.detail)

print("\n=== 2b. PII hard-block guardrail ===")
r_pii_block = check_pii_hard_block(manifest1)
check("hard-block guardrail correctly fails this manifest (contains an AWS key)", not r_pii_block.passed, r_pii_block.detail)

import copy as _copy
clean_manifest = _copy.deepcopy(manifest1)
clean_manifest.has_hard_block = False
r_pii_clean = check_pii_hard_block(clean_manifest)
check("hard-block guardrail passes when has_hard_block is False", r_pii_clean.passed, r_pii_clean.detail)

print("\n=== 2c. answer-groundedness guardrail (check_answer_guardrails) ===")
grounded_verdicts = [{"question": "q1", "verdict": {"grounded": True, "reason": "ok"}}]
ungrounded_verdicts = [{"question": "q1", "verdict": {"grounded": True, "reason": "ok"}},
                       {"question": "q2", "verdict": {"grounded": False, "reason": "made up a claim"}}]
r_grounded = check_answer_guardrails(grounded_verdicts)
r_ungrounded = check_answer_guardrails(ungrounded_verdicts)
check("answer guardrail passes when all verdicts are grounded", r_grounded.passed, r_grounded.detail)
check("answer guardrail fails when any verdict is ungrounded", not r_ungrounded.passed, r_ungrounded.detail)

record_manifest_history(manifest1)
check("manifest history file written", MANIFEST_HISTORY_PATH.exists())

print("\n=== 3. a second, identical run should still pass all structural gates ===")
manifest2 = build_staging_index(source_dir, run_id="run-2")
r1b = check_chunking_regression(manifest2)
r2b = check_embedding_model_drift(manifest2)
r3b = check_partial_reindex(manifest2, source_dir)
check("chunking regression check passes on identical rerun", r1b.passed, r1b.detail)
check("embedding drift check passes on identical rerun", r2b.passed, r2b.detail)
check("partial reindex check passes on identical rerun", r3b.passed, r3b.detail)
record_manifest_history(manifest2)

print("\n=== 4. simulated embedding model drift should be CAUGHT ===")
import copy
drifted = copy.deepcopy(manifest2)
drifted.embedding_model = "nomic-embed-text-v2"  # simulate a silent bump
r_drift = check_embedding_model_drift(drifted)
check("drifted embedding model is correctly rejected", not r_drift.passed, r_drift.detail)

print("\n=== 5. simulated chunking regression should be CAUGHT ===")
drifted_chunks = copy.deepcopy(manifest2)
drifted_chunks.chunk_count = manifest2.chunk_count * 3  # simulate a splitter regression
r_chunk = check_chunking_regression(drifted_chunks)
check("chunking blow-up is correctly rejected", not r_chunk.passed, r_chunk.detail)

print("\n=== 6. simulated partial re-index should be CAUGHT ===")
drifted_partial = copy.deepcopy(manifest2)
missing_doc = next(iter(drifted_partial.source_doc_hashes))
del drifted_partial.source_doc_hashes[missing_doc]
r_partial = check_partial_reindex(drifted_partial, source_dir)
check(f"missing doc ({missing_doc}) is correctly rejected", not r_partial.passed, r_partial.detail)

print("\n=== 7. retrieval quality gate runs end-to-end against fake RAGAS ===")
# check_retrieval_quality takes already-generated rows (question/contexts/
# ground_truth/answer); build the same shape here with a canned answer.
from src.evaluate import check_retrieval_quality

eval_path = os.path.join(PROJECT_DIR, "data", "eval_dataset.jsonl")
eval_rows = [json.loads(line) for line in open(eval_path) if line.strip()]
generated_rows = []
for row in eval_rows:
    hits = retrieve(row["question"], collection_name=STAGING_COLLECTION, k=4)
    generated_rows.append({
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "contexts": [h["text"] for h in hits],
        "answer": "canned smoketest answer",
    })
r_quality = check_retrieval_quality(generated_rows)
check("retrieval quality gate returns all 3 metrics", set(r_quality.metrics.keys()) == {"faithfulness", "context_precision", "context_recall"}, r_quality.metrics)
print(f"  (fake-judge scores -- not meaningful as real quality signal: {r_quality.metrics})")

print("\n=== 8. promotion path: promote_staging_to_prod ===")
promote_staging_to_prod()
client2 = get_chroma_client()
prod_names = [c.name for c in client2.list_collections()]
check("prod collection exists after promotion", PROD_COLLECTION in prod_names, f"got {prod_names}")

hits = retrieve("What caused the checkout latency spike?", collection_name=PROD_COLLECTION, k=3)
check("prod retrieval returns hits", len(hits) > 0, f"got {len(hits)}")
check("retrieved hits carry doc_id metadata", all("doc_id" in h["metadata"] for h in hits))

print("\n=== 9. src.query CLI path (ask()) triggers the DAG, doesn't call an LLM in-process ===")
# monkeypatch subprocess.run since there's no local Airflow webserver/
# scheduler in this offline smoketest
import subprocess
import src.query as query_mod

captured = {}
class _FakeCompletedProcess:
    returncode = 0
    stdout = "Triggered dag_run"
    stderr = ""

def _fake_run(cmd, **kwargs):
    captured["cmd"] = cmd
    return _FakeCompletedProcess()

_real_run = subprocess.run
subprocess.run = _fake_run
try:
    query_mod.ask("What caused the checkout latency spike?")
    check("src.query.ask() ran without raising", True)
    check(
        "src.query.ask() invoked `airflow dags trigger postmortem_query_pipeline`",
        captured.get("cmd", [])[:4] == ["airflow", "dags", "trigger", "postmortem_query_pipeline"],
        captured.get("cmd"),
    )
    check(
        "src.query.ask() passed the question through --conf as JSON",
        json.loads(captured["cmd"][-1])["question"] == "What caused the checkout latency spike?",
        captured.get("cmd"),
    )
except Exception as e:
    check("src.query.ask() ran without raising", False, repr(e))
finally:
    subprocess.run = _real_run

print("\n=== 10. DAG files import and wire up under fake airflow.sdk ===")
dag_path = os.path.join(PROJECT_DIR, "dags")
sys.path.insert(0, dag_path)
try:
    import postmortem_rag_pipeline  # noqa: F401
    check("dags/postmortem_rag_pipeline.py imports cleanly", True)
except Exception as e:
    check("dags/postmortem_rag_pipeline.py imports cleanly", False, repr(e))

try:
    import postmortem_query_pipeline  # noqa: F401
    check("dags/postmortem_query_pipeline.py imports cleanly", True)
except Exception as e:
    check("dags/postmortem_query_pipeline.py imports cleanly", False, repr(e))

shutil.rmtree(CHROMA_TMP, ignore_errors=True)

print(f"\n=== RESULT: {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    print("FAILED CHECKS:", FAIL)
    sys.exit(1)
sys.exit(0)

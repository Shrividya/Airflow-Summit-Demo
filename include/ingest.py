"""Ingestion logic: chunking, embedding, and Chroma staging/prod, called from
dags/postmortem_rag_pipeline.py's @task wrappers and from the query path.
Source documents are scanned for PII/secrets (include/guardrails.py) before chunking."""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from include.guardrails import scan_and_redact, has_hard_finding
from huggingface_hub import InferenceClient
import chromadb

EMBEDDING_MODEL = os.environ.get("PM_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.environ.get("HF_TOKEN")
CHUNK_SIZE = int(os.environ.get("PM_RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("PM_RAG_CHUNK_OVERLAP", "120"))
CHROMA_PATH = os.environ.get("PM_RAG_CHROMA_PATH", "/tmp/postmortem-rag-chroma")

STAGING_COLLECTION = "postmortem_index_staging"
PROD_COLLECTION = "postmortem_index_prod"


@dataclass
class IngestManifest:
    """Metadata for one ingestion run, persisted and diffed against the
    previous run by the quality-gate tasks in the DAG."""
    run_id: str
    source_doc_count: int
    chunk_count: int
    avg_chunk_chars: float
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    source_doc_hashes: dict
    created_at: str
    pii_findings: dict
    has_hard_block: bool


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware character splitter -- no langchain dependency.
    Swap for a real text splitter in production."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) > chunk_size:
                # hard-split an oversized paragraph
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end])
                    start = end - overlap
            else:
                buf = para
    if buf:
        chunks.append(buf)
    return chunks


def _hash_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_source_documents(source_dir: str) -> dict[str, str]:
    """Return {doc_id: raw_text} for every postmortem markdown file."""
    docs = {}
    for path in sorted(glob.glob(os.path.join(source_dir, "*.md"))):
        doc_id = Path(path).stem
        docs[doc_id] = Path(path).read_text(encoding="utf-8")
    return docs


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """`input_type` is unused by all-MiniLM-L6-v2, kept for parity with
    asymmetric embedding APIs."""
    if not texts:
        raise ValueError("embed_texts called with an empty list of texts")
    client = InferenceClient(model=EMBEDDING_MODEL, token=HF_TOKEN)
    embeddings = client.feature_extraction(texts)
    return [list(map(float, row)) for row in embeddings]


def get_chroma_client(path: str = CHROMA_PATH) -> "chromadb.ClientAPI":
    return chromadb.PersistentClient(path=path)


def _load_last_successful_manifest() -> Optional[dict]:
    """Mirrors include/evaluate.py's _load_previous_manifest (kept separate to
    avoid a circular import). manifest_history.json is only appended to after
    all quality gates pass, so this is always a known-good, promoted-to-prod
    state -- safe to diff against and to copy reused chunks out of PROD_COLLECTION."""
    history_path = Path(CHROMA_PATH) / "manifest_history.json"
    if not history_path.exists():
        return None
    history = json.loads(history_path.read_text())
    return history[-1] if history else None


def plan_incremental_ingest(source_dir: str, run_id: Optional[str] = None) -> dict:
    """Diff current source docs against the last successfully-promoted manifest
    so only new/changed postmortems need re-chunking and re-embedding; everything
    else can be copied from PROD_COLLECTION as-is in assemble_staging_index. An
    embedding-model change forces a full re-embed since old vectors aren't
    comparable to new ones."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    docs = load_source_documents(source_dir)
    if not docs:
        raise ValueError(f"No source documents (*.md) found in {source_dir!r}. Check PM_RAG_SOURCE_DIR / SOURCE_DIR.")

    # hash pre-redaction so the partial-reindex gate ignores redaction noise
    current_hashes = {doc_id: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] for doc_id, text in docs.items()}
    prev = _load_last_successful_manifest()
    prev_hashes = prev["source_doc_hashes"] if prev else {}
    # embedding model OR chunking params changing invalidates every existing vector/chunk
    # boundary, not just the changed docs' -- force a full re-embed rather than silently
    # mixing old and new chunk boundaries across reused vs. re-embedded docs.
    force_full_reembed = bool(prev) and (
        prev["embedding_model"] != EMBEDDING_MODEL
        or prev["chunk_size"] != CHUNK_SIZE
        or prev["chunk_overlap"] != CHUNK_OVERLAP
    )

    to_embed = [doc_id for doc_id, h in current_hashes.items() if force_full_reembed or prev_hashes.get(doc_id) != h]
    reused = [doc_id for doc_id in current_hashes if doc_id not in to_embed]
    removed = sorted(set(prev_hashes) - set(current_hashes))

    return {
        "run_id": run_id,
        "to_embed": sorted(to_embed),
        "reused": sorted(reused),
        "removed": removed,
        "current_hashes": current_hashes,
    }


def embed_document(source_dir: str, doc_id: str, run_id: str) -> dict:
    """Chunk + embed a single postmortem. Meant to be mapped over
    plan['to_embed'] (see plan_incremental_ingest) so unrelated, unchanged
    docs don't pay for each other's embedding API round-trips."""
    raw_text = Path(os.path.join(source_dir, f"{doc_id}.md")).read_text(encoding="utf-8")
    text, findings = scan_and_redact(raw_text)
    chunks = _chunk_text(text)
    embeddings = embed_texts(chunks, input_type="document")
    ids = [f"{doc_id}::chunk-{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "chunk_index": i, "embedding_model": EMBEDDING_MODEL, "run_id": run_id}
        for i in range(len(chunks))
    ]
    return {
        "doc_id": doc_id,
        "ids": ids,
        "documents": chunks,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "pii_findings": findings,
        "has_hard_block": has_hard_finding(findings),
    }


def assemble_staging_index(plan: dict, embedded_docs: list[dict]) -> IngestManifest:
    """Build STAGING_COLLECTION from this run's freshly-embedded docs
    (plan['to_embed'], from mapped embed_document calls) plus chunks copied
    straight out of PROD_COLLECTION for everything unchanged (plan['reused'])."""
    client = get_chroma_client()
    if STAGING_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(STAGING_COLLECTION)
    collection = client.create_collection(
        STAGING_COLLECTION,
        metadata={"embedding_model": EMBEDDING_MODEL, "run_id": plan["run_id"]},
    )

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_embeddings: list[list[float]] = []
    all_metadatas: list[dict] = []
    pii_findings: dict[str, list] = {}
    any_hard_block = False

    for item in embedded_docs:
        all_ids.extend(item["ids"])
        all_docs.extend(item["documents"])
        all_embeddings.extend(item["embeddings"])
        all_metadatas.extend(item["metadatas"])
        if item["pii_findings"]:
            pii_findings[item["doc_id"]] = item["pii_findings"]
        if item["has_hard_block"]:
            any_hard_block = True

    if plan["reused"]:
        if PROD_COLLECTION not in [c.name for c in client.list_collections()]:
            raise RuntimeError(
                f"Plan expects {len(plan['reused'])} doc(s) reused from {PROD_COLLECTION!r}, but no prod "
                "collection exists yet -- delete manifest_history.json to force a full re-index."
            )
        prod = client.get_collection(PROD_COLLECTION)
        reused_set = set(plan["reused"])
        found_doc_ids: set = set()
        prod_data = prod.get(include=["documents", "embeddings", "metadatas"])
        for chunk_id, doc, meta, embedding in zip(
            prod_data["ids"], prod_data["documents"], prod_data["metadatas"], prod_data["embeddings"]
        ):
            if meta["doc_id"] in reused_set:
                found_doc_ids.add(meta["doc_id"])
                all_ids.append(chunk_id)
                all_docs.append(doc)
                all_embeddings.append(list(map(float, embedding)))
                all_metadatas.append(meta)

        missing = reused_set - found_doc_ids
        if missing:
            # prod is out of sync with the manifest we diffed against (e.g. a prior
            # promote_to_prod failed after record_manifest_history succeeded) -- fail
            # loudly instead of silently shipping a staging index missing these docs.
            raise RuntimeError(
                f"{PROD_COLLECTION!r} has no chunks for reused doc(s) {sorted(missing)}, but the last "
                "successful manifest expected them unchanged -- delete manifest_history.json to force "
                "a full re-index."
            )

    if all_ids:
        collection.add(ids=all_ids, documents=all_docs, embeddings=all_embeddings, metadatas=all_metadatas)

    manifest = IngestManifest(
        run_id=plan["run_id"],
        source_doc_count=len(plan["to_embed"]) + len(plan["reused"]),
        chunk_count=len(all_ids),
        avg_chunk_chars=sum(len(d) for d in all_docs) / max(len(all_docs), 1),
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        source_doc_hashes=plan["current_hashes"],
        created_at=datetime.now(timezone.utc).isoformat(),
        pii_findings=pii_findings,
        has_hard_block=any_hard_block,
    )

    manifest_path = Path(CHROMA_PATH) / "last_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    return manifest


def build_staging_index(source_dir: str, run_id: Optional[str] = None) -> IngestManifest:
    """Full rebuild: chunk + embed every postmortem into STAGING, ignoring any
    previous manifest. Kept for the first-ever run and for callers (tests, CLI)
    that want a from-scratch index; the DAG uses the incremental
    plan_incremental_ingest / embed_document / assemble_staging_index path so
    routine runs only re-embed docs that actually changed."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    docs = load_source_documents(source_dir)
    if not docs:
        raise ValueError(f"No source documents (*.md) found in {source_dir!r}. Check PM_RAG_SOURCE_DIR / SOURCE_DIR.")

    plan = {
        "run_id": run_id,
        "to_embed": sorted(docs.keys()),
        "reused": [],
        "removed": [],
        "current_hashes": {doc_id: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] for doc_id, text in docs.items()},
    }
    embedded_docs = [embed_document(source_dir, doc_id, run_id) for doc_id in plan["to_embed"]]
    return assemble_staging_index(plan, embedded_docs)


def promote_staging_to_prod() -> None:
    """Copy the staging collection into the prod collection. Only place prod is written."""
    client = get_chroma_client()
    staging = client.get_collection(STAGING_COLLECTION)
    existing = [c.name for c in client.list_collections()]
    if PROD_COLLECTION in existing:
        client.delete_collection(PROD_COLLECTION)

    data = staging.get(include=["documents", "embeddings", "metadatas"])
    prod = client.create_collection(PROD_COLLECTION, metadata=staging.metadata)
    if data["ids"]:
        prod.add(
            ids=data["ids"],
            documents=data["documents"],
            embeddings=data["embeddings"],
            metadatas=data["metadatas"],
        )


def retrieve(query: str, collection_name: str = PROD_COLLECTION, k: int = 4) -> list[dict]:
    """Retrieve top-k chunks for a query from the given collection."""
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    query_embedding = embed_texts([query], input_type="query")[0]
    result = collection.query(query_embeddings=[query_embedding], n_results=k)
    hits = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits

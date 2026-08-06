"""Ingestion logic for the postmortem RAG pipeline. Plain functions, called
from dags/postmortem_rag_pipeline.py's @task-decorated wrappers and from
the query path, so it's testable and reusable outside Airflow.

Chroma is local/on-disk (swap `get_chroma_client` for a hosted client in
production). Embeddings use a local model served by Ollama (default
`nomic-embed-text`) via its OpenAI-compatible endpoint; the model name is
recorded on every chunk/collection for the embedding-model-drift gate.

Source documents are scanned for PII/secrets (src/guardrails.py) and
redacted before chunking, so raw sensitive strings never reach the
embedding API or the vector index.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb

EMBEDDING_MODEL = os.environ.get("PM_RAG_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
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
    """Simple paragraph-aware character splitter, dependency-light so this
    runs without langchain. Swap for RecursiveCharacterTextSplitter or a
    semantic chunker in production."""
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
    """`input_type` ("document"/"query") is unused by nomic-embed-text but
    kept in the signature for parity with asymmetric embedding APIs."""
    if not texts:
        raise ValueError("embed_texts called with an empty list of texts")

    import openai

    client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def get_chroma_client(path: str = CHROMA_PATH) -> "chromadb.ClientAPI":
    return chromadb.PersistentClient(path=path)


def build_staging_index(source_dir: str, run_id: Optional[str] = None) -> IngestManifest:
    """Chunk + embed all postmortems into the STAGING collection. Nothing
    here is queryable in prod until the DAG's quality gate promotes
    staging -> prod."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    docs = load_source_documents(source_dir)
    if not docs:
        raise ValueError(
            f"No source documents (*.md) found in {source_dir!r}. "
            "Check PM_RAG_SOURCE_DIR / SOURCE_DIR -- refusing to build an "
            "index from zero documents."
        )

    client = get_chroma_client()
    client.delete_collection(STAGING_COLLECTION) if STAGING_COLLECTION in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(
        STAGING_COLLECTION,
        metadata={"embedding_model": EMBEDDING_MODEL, "run_id": run_id},
    )

    from src.guardrails import scan_and_redact, has_hard_finding

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []
    doc_hashes: dict[str, str] = {}
    pii_findings: dict[str, list] = {}
    any_hard_block = False

    for doc_id, raw_text in docs.items():
        # hash the raw text (pre-redaction) so the partial-reindex gate
        # detects real content changes, not redaction noise
        doc_hashes[doc_id] = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
        text, findings = scan_and_redact(raw_text)
        if findings:
            pii_findings[doc_id] = findings
            if has_hard_finding(findings):
                any_hard_block = True
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_id}::chunk-{i}")
            all_metadatas.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "embedding_model": EMBEDDING_MODEL,
                "run_id": run_id,
            })

    embeddings = embed_texts(all_chunks, input_type="document")
    collection.add(ids=all_ids, documents=all_chunks, embeddings=embeddings, metadatas=all_metadatas)

    manifest = IngestManifest(
        run_id=run_id,
        source_doc_count=len(docs),
        chunk_count=len(all_chunks),
        avg_chunk_chars=sum(len(c) for c in all_chunks) / max(len(all_chunks), 1),
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        source_doc_hashes=doc_hashes,
        created_at=datetime.now(timezone.utc).isoformat(),
        pii_findings=pii_findings,
        has_hard_block=any_hard_block,
    )

    manifest_path = Path(CHROMA_PATH) / "last_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    return manifest


def promote_staging_to_prod() -> None:
    """Copy the staging collection into the prod collection. Only called
    by the DAG after every quality gate has passed; the only place prod is
    ever written."""
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

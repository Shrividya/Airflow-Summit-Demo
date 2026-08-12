
"""Minimal stand-in for huggingface_hub -- just enough of InferenceClient's
feature_extraction to exercise include/ingest.py's embed_texts (deterministic
hash-based vectors, enough for the fake chromadb's cosine-similarity ranking
to be meaningful)."""
import hashlib


def _fake_embed(text: str, dims: int = 32) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # repeat/truncate hash bytes to `dims` floats in [-1, 1]
    return [(h[i % len(h)] / 127.5) - 1.0 for i in range(dims)]


class InferenceClient:
    def __init__(self, *a, **kw):
        pass

    def feature_extraction(self, texts, **kw):
        if isinstance(texts, str):
            texts = [texts]
        return [_fake_embed(t) for t in texts]

"""Minimal stand-in for the voyageai package. Embeddings are deterministic
hash-based vectors, just enough for cosine-similarity ranking in the fake
chromadb to be meaningful for testing branching logic."""
import hashlib


def _fake_embed(text: str, dims: int = 32) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # repeat/truncate hash bytes to `dims` floats in [-1, 1]
    vals = [(h[i % len(h)] / 127.5) - 1.0 for i in range(dims)]
    return vals


class _EmbedResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class Client:
    def __init__(self, *a, **kw):
        pass

    def embed(self, texts, model, input_type=None, **kw):
        return _EmbedResult([_fake_embed(t) for t in texts])

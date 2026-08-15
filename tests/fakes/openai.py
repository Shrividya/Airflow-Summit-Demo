"""Minimal stand-in for the openai package -- chat completions and
embeddings (deterministic hash-based vectors, enough for the fake
chromadb's cosine-similarity ranking to be meaningful)."""
import hashlib


def _fake_embed(text: str, dims: int = 32) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # repeat/truncate hash bytes to `dims` floats in [-1, 1]
    return [(h[i % len(h)] / 127.5) - 1.0 for i in range(dims)]


class _Embedding:
    def __init__(self, embedding):
        self.embedding = embedding


class _EmbeddingsResponse:
    def __init__(self, data):
        self.data = data


class _Embeddings:
    def create(self, model, input, **kw):  # noqa: ARG002
        return _EmbeddingsResponse([_Embedding(_fake_embed(t)) for t in input])


class _TextBlock:
    def __init__(self, text):
        self.content = text


class _Choice:
    def __init__(self, text):
        self.message = _TextBlock(text)


class _ChatCompletion:
    def __init__(self, text):
        self.choices = [_Choice(text)]


class _Completions:
    def create(self, model, messages, **kw):  # noqa: ARG002
        prompt = messages[-1]["content"]
        return _ChatCompletion(f"[fake-answer grounded in {len(prompt)} chars of context]")


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class OpenAI:
    def __init__(self, *a, **kw):  # noqa: ARG002
        self.embeddings = _Embeddings()
        self.chat = _Chat()

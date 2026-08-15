"""Minimal in-memory stand-in for chromadb. Not a real similarity search --
good enough for testing branching/manifest logic, not retrieval quality."""
import math


class _Collection:
    def __init__(self, name, metadata=None):
        self.name = name
        self.metadata = metadata or {}
        self._ids = []
        self._documents = []
        self._embeddings = []
        self._metadatas = []

    def add(self, ids, documents, embeddings, metadatas):
        self._ids.extend(ids)
        self._documents.extend(documents)
        self._embeddings.extend(embeddings)
        self._metadatas.extend(metadatas)

    def get(self, include=None):  # noqa: ARG002
        return {
            "ids": self._ids,
            "documents": self._documents,
            "embeddings": self._embeddings,
            "metadatas": self._metadatas,
        }

    def query(self, query_embeddings, n_results=4):
        q = query_embeddings[0]

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = math.sqrt(sum(x * x for x in a)) or 1e-9
            nb = math.sqrt(sum(y * y for y in b)) or 1e-9
            return dot / (na * nb)

        scored = sorted(
            zip(self._ids, self._documents, self._metadatas, self._embeddings, strict=True),
            key=lambda row: -cosine(q, row[3]),
        )[:n_results]
        ids = [r[0] for r in scored]
        docs = [r[1] for r in scored]
        metas = [r[2] for r in scored]
        dists = [1 - cosine(q, r[3]) for r in scored]
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}


class PersistentClient:
    _stores = {}  # keyed by path so repeated PersistentClient(path=...) calls share state

    def __init__(self, path):
        self.path = path
        if path not in PersistentClient._stores:
            PersistentClient._stores[path] = {}
        self._store = PersistentClient._stores[path]

    def list_collections(self):
        class _Name:
            def __init__(self, name):
                self.name = name
        return [_Name(n) for n in self._store]

    def create_collection(self, name, metadata=None):
        col = _Collection(name, metadata)
        self._store[name] = col
        return col

    def get_collection(self, name):
        return self._store[name]

    def delete_collection(self, name):
        self._store.pop(name, None)

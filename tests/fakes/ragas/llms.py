"""Fake `ragas.llms.llm_factory`: returns a placeholder so src/evaluate.py's
call pattern is exercised offline. The fake `evaluate()` ignores the llm=
kwarg and scores via the word-overlap heuristic in ragas/__init__.py."""


class _FakeRagasLLM:
    def __init__(self, model, provider, client):
        self.model = model
        self.provider = provider
        self.client = client


def llm_factory(model, provider=None, client=None):
    return _FakeRagasLLM(model, provider, client)

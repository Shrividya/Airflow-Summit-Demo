"""Fake `ragas.llms.base.InstructorLLM`: returns a placeholder so
include/evaluate.py's call pattern is exercised offline. The fake `evaluate()`
in ragas/__init__.py ignores the llm= kwarg and scores via the word-overlap
heuristic instead of actually calling out to `client`."""


class _FakeRagasLLM:
    def __init__(self, client, model, provider, model_args=None, cache=None, **kwargs):  # noqa: ARG002
        self.client = client
        self.model = model
        self.provider = provider
        self.model_args = model_args if model_args is not None else {}
        self.cache = cache


def InstructorLLM(client, model, provider, model_args=None, cache=None, **kwargs):
    return _FakeRagasLLM(client, model, provider, model_args=model_args, cache=cache, **kwargs)

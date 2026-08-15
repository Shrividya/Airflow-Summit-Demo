"""Minimal stand-in for `instructor` -- the real package's `from_openai`
transitively imports real `openai.types.chat` submodules that the fake
`openai` module (tests/fakes/openai.py) doesn't provide, and structured-
output patching isn't exercised by this offline smoke test anyway (the fake
`ragas.evaluate` in tests/fakes/ragas/__init__.py never calls the patched
client, and the fake `ragas.llms.base.InstructorLLM` just stores it). So
`from_openai` here is a no-op that returns the client unchanged."""


class Mode:
    JSON = "json"
    JSON_SCHEMA = "json_schema"
    TOOLS = "tools"


def from_openai(client, mode=None, **kwargs):  # noqa: ARG001
    return client

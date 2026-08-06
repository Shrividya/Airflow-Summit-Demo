"""Minimal stand-in for the anthropic package. Generation is a canned,
deterministic string so the RAG generation step can be exercised offline."""


class _TextBlock:
    def __init__(self, text):
        self.text = text


class _Message:
    def __init__(self, text):
        self.content = [_TextBlock(text)]


class _Messages:
    def create(self, model, messages, max_tokens=1024, **kw):
        prompt = messages[0]["content"]
        context_len = len(prompt)
        return _Message(f"[fake-answer grounded in {context_len} chars of context]")


class Anthropic:
    def __init__(self, *a, **kw):
        self.messages = _Messages()

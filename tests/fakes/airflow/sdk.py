"""Minimal stand-in for airflow.sdk -- just enough to import-test the two
DAG files' structure (decorators, outlets/params, task graph wiring
including .expand() and @task.llm/@task.branch). Doesn't execute task
bodies; those are tested directly against include/ingest.py etc. elsewhere in
this smoketest.
"""


class Asset:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Asset({self.name!r})"


class Param:
    def __init__(self, default=None, **kwargs):
        self.default = default
        self.kwargs = kwargs


class _XComArg:
    """Stand-in for what a called @task-decorated function returns in real
    Airflow: a lazy reference resolved at run time, not the result."""
    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __rshift__(self, other):
        return other

    def __rrshift__(self, other):
        return self


class _TaskWrapper:
    """Supports both `wrapper(...)` and `wrapper.expand(...)`, matching
    real TaskFlow's API surface closely enough to import-test wiring."""
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "task")

    def __call__(self, *a, **kw):
        return _XComArg(self.fn, a, kw)

    def expand(self, **kw):
        return _XComArg(self.fn, (), kw)


def _wrap(fn):
    wrapper = _TaskWrapper(fn)
    return wrapper


def task(*dargs, **dkwargs):
    # supports both @task and @task(outlets=[...], retries=..., etc.)
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return _wrap(dargs[0])

    def decorator(fn):
        return _wrap(fn)
    return decorator


def _branch(*dargs, **dkwargs):
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return _wrap(dargs[0])

    def decorator(fn):
        return _wrap(fn)
    return decorator


def _llm(*dargs, **dkwargs):
    # only import-time wiring is under test; real LLM calls aren't simulated
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return _wrap(dargs[0])

    def decorator(fn):
        return _wrap(fn)
    return decorator


task.branch = _branch
task.llm = _llm


class DAG:
    def __init__(self, dag_id, **kwargs):
        self.dag_id = dag_id
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

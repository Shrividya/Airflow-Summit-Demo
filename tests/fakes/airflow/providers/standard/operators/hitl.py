"""Minimal stand-in for airflow.providers.standard.operators.hitl -- just
enough to import-test dags/postmortem_rag_pipeline.py's task graph wiring
(HITLOperator is only ever constructed, wired with `>>`, and has `.output`
passed along to a downstream @task.branch -- never executed -- in this
offline smoke test)."""


class HITLOperator:
    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs
        self.output = f"XComArg({task_id})"

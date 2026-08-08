"""Minimal stand-in for airflow.providers.standard.operators.hitl -- just
enough to import-test dags/postmortem_rag_pipeline.py's task graph wiring
(ApprovalOperator is only ever constructed and wired with `>>`, never
executed, in this offline smoke test)."""


class ApprovalOperator:
    def __init__(self, task_id, **kwargs):
        self.task_id = task_id
        self.kwargs = kwargs

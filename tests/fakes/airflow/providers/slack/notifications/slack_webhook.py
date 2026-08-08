"""Minimal stand-in for airflow.providers.slack.notifications.slack_webhook --
just enough to import-test dags/postmortem_rag_pipeline.py's task graph
wiring (SlackWebhookNotifier is only ever constructed and passed as a
`notifiers=` kwarg, never executed, in this offline smoke test)."""


class SlackWebhookNotifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

"""CLI: trigger the postmortem_query_pipeline DAG for a question.

    python -m src.query "What caused the checkout latency spike?"

Answering a question is a DAG run against the PROD collection, not a plain
Python call -- watch the answer in the Airflow UI (Grid -> this run ->
deliver_answer task's logs) or:

    airflow tasks logs postmortem_query_pipeline <run_id> deliver_answer
"""
from __future__ import annotations

import json
import subprocess
import sys


def ask(question: str) -> None:
    conf = json.dumps({"question": question})
    result = subprocess.run(
        ["airflow", "dags", "trigger", "postmortem_query_pipeline", "--conf", conf],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    print(
        "\nTriggered postmortem_query_pipeline. Check the run's task logs in the "
        "Airflow UI (Grid view) -- 'deliver_answer' if the question and answer "
        "both passed their guardrails, 'refuse' if the input-safety guardrail "
        "blocked the question, or 'block_answer' if the groundedness guardrail "
        "blocked the generated answer."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.query "your question"')
        sys.exit(1)
    ask(" ".join(sys.argv[1:]))

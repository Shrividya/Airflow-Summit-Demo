"""Unit tests for dags/postmortem_query_pipeline.py's `_record_result`, the
SQLite writer streamlit_app.py polls for an answer. Run inside a container
with the real Airflow deps installed (not part of the offline fake-based
smoketest).
"""
import json
import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch


def test_record_result_writes_row():
    from dags.postmortem_query_pipeline import _record_result

    with tempfile.TemporaryDirectory() as tmp:
        fake_context = {"dag_run": MagicMock(run_id="run-1")}
        with (
            patch.dict(os.environ, {"AIRFLOW_HOME": tmp}),
            patch("dags.postmortem_query_pipeline.get_current_context", return_value=fake_context),
        ):
            _record_result("answered", "the answer")

        conn = sqlite3.connect(os.path.join(tmp, "include", "query_results.db"))
        row = conn.execute(
            "SELECT status, text, sources_json FROM query_results WHERE run_id = ?", ("run-1",)
        ).fetchone()
        conn.close()
        assert row == ("answered", "the answer", None)


def test_record_result_stores_sources_as_json():
    from dags.postmortem_query_pipeline import _record_result

    sources = [{"doc_id": "INC-1", "chunk_index": 0, "distance": 0.123}]
    with tempfile.TemporaryDirectory() as tmp:
        fake_context = {"dag_run": MagicMock(run_id="run-1")}
        with (
            patch.dict(os.environ, {"AIRFLOW_HOME": tmp}),
            patch("dags.postmortem_query_pipeline.get_current_context", return_value=fake_context),
        ):
            _record_result("answered", "the answer", sources=sources)

        conn = sqlite3.connect(os.path.join(tmp, "include", "query_results.db"))
        row = conn.execute(
            "SELECT status, text, sources_json FROM query_results WHERE run_id = ?", ("run-1",)
        ).fetchone()
        conn.close()
        assert row[:2] == ("answered", "the answer")
        assert json.loads(row[2]) == sources


def test_record_result_overwrites_same_run_id():
    from dags.postmortem_query_pipeline import _record_result

    with tempfile.TemporaryDirectory() as tmp:
        fake_context = {"dag_run": MagicMock(run_id="run-1")}
        with (
            patch.dict(os.environ, {"AIRFLOW_HOME": tmp}),
            patch("dags.postmortem_query_pipeline.get_current_context", return_value=fake_context),
        ):
            _record_result("refused", "first")
            _record_result("answered", "second")

        conn = sqlite3.connect(os.path.join(tmp, "include", "query_results.db"))
        rows = conn.execute("SELECT status, text FROM query_results WHERE run_id = ?", ("run-1",)).fetchall()
        conn.close()
        assert rows == [("answered", "second")]

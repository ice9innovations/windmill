import logging
import os
import sys

import pytest

_WORKERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workers")
if _WORKERS_DIR not in sys.path:
    sys.path.insert(0, _WORKERS_DIR)

from workers.content_analysis_worker import ContentAnalysisWorker
from workers.noun_consensus_worker import NounConsensusWorker


class _Cursor:
    def __init__(self):
        self.sql = None
        self.params = None
        self.closed = False

    def execute(self, sql, params):
        self.sql = sql
        self.params = params

    def close(self):
        self.closed = True


class _Connection:
    autocommit = True
    closed = 0

    def __init__(self):
        self.cursor_instance = _Cursor()

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        pass


@pytest.mark.parametrize(
    ("event_type", "event_insert_enabled"),
    [("completed", False), ("failed", True)],
)
def test_specialized_terminal_result_obeys_failed_only_policy(
    event_type, event_insert_enabled
):
    worker = ContentAnalysisWorker.__new__(ContentAnalysisWorker)
    worker.db_conn = _Connection()
    worker.worker_id = "worker_test"
    worker.service_events_mode = "failed"
    worker.logger = logging.getLogger("test_specialized_service_event_policy")

    worker._persist_terminal_result(
        image_id=123,
        payload={"status": event_type},
        processing_time=0.25,
        event_type=event_type,
        source_service="test",
        event_data={"detail": "test"},
        status="success" if event_type == "completed" else "failed",
    )

    cursor = worker.db_conn.cursor_instance
    assert "INSERT INTO results" in cursor.sql
    assert "INSERT INTO service_events" in cursor.sql
    assert "WHERE %s" in cursor.sql
    assert cursor.params[-1] is event_insert_enabled
    assert cursor.closed is True


def _noun_consensus_worker():
    worker = NounConsensusWorker.__new__(NounConsensusWorker)
    worker.db_conn = _Connection()
    worker.worker_id = "worker_test"
    worker.service_events_mode = "failed"
    worker.logger = logging.getLogger("test_specialized_service_event_policy")
    return worker


def test_combined_consensus_event_payload_is_cast_to_jsonb():
    worker = _noun_consensus_worker()

    worker._persist_terminal_consensus_results(
        image_id=123,
        noun_payload={"status": "success", "services_present": ["blip"]},
        verb_payload={"status": "success", "services_present": ["blip"]},
        processing_time=0.25,
        source_service="blip",
    )

    cursor = worker.db_conn.cursor_instance
    assert "data::jsonb" in cursor.sql
    assert cursor.params[-1] is False
    assert cursor.closed is True


def test_single_consensus_event_payload_is_cast_to_jsonb():
    worker = _noun_consensus_worker()

    worker._persist_terminal_noun_consensus_result(
        image_id=123,
        payload={"status": "success"},
        processing_time=0.25,
        source_service="blip",
        event_data={"services_present": ["blip"]},
    )

    cursor = worker.db_conn.cursor_instance
    assert "%s::jsonb" in cursor.sql
    assert cursor.params[-1] is False
    assert cursor.closed is True

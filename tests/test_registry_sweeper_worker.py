import os
import sys

import pytest

_WORKERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workers")
if _WORKERS_DIR not in sys.path:
    sys.path.insert(0, _WORKERS_DIR)

from workers.registry_sweeper_worker import _prune_service_events


class _Cursor:
    def __init__(self, rowcount=0):
        self.rowcount = rowcount
        self.executed = None
        self.closed = False

    def execute(self, sql, params):
        self.executed = (sql, params)

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def test_prune_service_events_deletes_one_bounded_batch():
    cursor = _Cursor(rowcount=73)

    deleted = _prune_service_events(
        _Connection(cursor),
        retention_days=14,
        batch_size=10_000,
    )

    assert deleted == 73
    assert cursor.executed[1] == (14, 10_000)
    assert "ORDER BY created_at, event_id" in cursor.executed[0]
    assert "LIMIT %s" in cursor.executed[0]
    assert "FOR UPDATE SKIP LOCKED" in cursor.executed[0]
    assert cursor.closed is True


@pytest.mark.parametrize(
    ("retention_days", "batch_size", "message"),
    [
        (0, 10_000, "SERVICE_EVENTS_RETENTION_DAYS"),
        (14, 0, "SERVICE_EVENTS_RETENTION_BATCH_SIZE"),
    ],
)
def test_prune_service_events_rejects_unsafe_configuration(
    retention_days, batch_size, message
):
    with pytest.raises(ValueError, match=message):
        _prune_service_events(
            _Connection(_Cursor()),
            retention_days=retention_days,
            batch_size=batch_size,
        )

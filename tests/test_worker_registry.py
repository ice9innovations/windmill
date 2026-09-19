import logging

from core.worker_registry import ManagedWorkerRegistry


class _Cursor:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.executed = None
        self.closed = False

    def execute(self, sql, params):
        if self.fail:
            raise RuntimeError("database unavailable")
        self.executed = (sql, params)

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, *, fail=False):
        self.cursor_instance = _Cursor(fail=fail)
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.closed = True


def _registry(connection_factory):
    return ManagedWorkerRegistry(
        connection_factory=connection_factory,
        logger=logging.getLogger("test_worker_registry"),
        worker_id="worker_colors_123",
        service="colors",
        heartbeat_interval=30,
        host="test-host",
    )


def test_heartbeat_once_leases_and_closes_connection():
    connections = []

    def connection_factory(*, autocommit):
        assert autocommit is True
        connection = _Connection()
        connections.append(connection)
        return connection

    registry = _registry(connection_factory)

    assert registry._heartbeat_once() is True
    assert len(connections) == 1
    assert connections[0].cursor_instance.executed[1] == ("worker_colors_123",)
    assert connections[0].cursor_instance.closed is True
    assert connections[0].closed is True


def test_failed_heartbeat_also_closes_connection():
    connection = _Connection(fail=True)
    registry = _registry(lambda *, autocommit: connection)

    assert registry._heartbeat_once() is False
    assert connection.cursor_instance.closed is True
    assert connection.closed is True


def test_heartbeat_stagger_is_stable_and_within_interval():
    registry = _registry(lambda *, autocommit: _Connection())

    delay = registry._heartbeat_initial_delay()

    assert delay == registry._heartbeat_initial_delay()
    assert 0 <= delay < registry.heartbeat_interval

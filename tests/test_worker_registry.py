import logging

from core.worker_registry import ManagedWorkerRegistry


class _Cursor:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.executed = None
        self.statements = []
        self.closed = False

    def execute(self, sql, params):
        if self.fail:
            raise RuntimeError("database unavailable")
        self.executed = (sql, params)
        self.statements.append((sql, params))

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self, *, fail=False):
        self.cursor_instance = _Cursor(fail=fail)
        self.closed = False
        self.commit_count = 0

    def cursor(self):
        return self.cursor_instance

    def close(self):
        self.closed = True

    def commit(self):
        self.commit_count += 1


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


def test_registration_uses_fresh_connection_without_global_stale_sweep():
    connection = _Connection()
    registry = _registry(lambda *, autocommit: connection)

    registry._register_with_fresh_connection()

    sql = "\n".join(statement for statement, _ in connection.cursor_instance.statements)
    assert "WHERE service = %s AND host = %s" in sql
    assert "INSERT INTO worker_registry" in sql
    assert "last_heartbeat <" not in sql
    assert connection.cursor_instance.closed is True
    assert connection.closed is True


def test_failed_registration_closes_isolated_connection():
    connection = _Connection(fail=True)
    registry = _registry(lambda *, autocommit: connection)

    try:
        registry._register_with_fresh_connection()
    except RuntimeError:
        pass
    else:
        raise AssertionError("registration failure was not propagated")

    assert connection.cursor_instance.closed is True
    assert connection.closed is True


def test_start_retries_registration_before_starting_heartbeat(monkeypatch):
    registry = _registry(lambda *, autocommit: _Connection())
    registry.registration_retry_seconds = 0
    attempts = []

    def register():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("registration conflict")

    class _Thread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False

        def start(self):
            self.started = True

    monkeypatch.setattr(registry, "_register_with_fresh_connection", register)
    monkeypatch.setattr("core.worker_registry.threading.Thread", _Thread)

    registry.start()

    assert len(attempts) == 3
    assert registry._thread.started is True

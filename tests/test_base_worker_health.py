"""Tests for BaseWorker's animal-farm service health-check and self-eviction logic.

BaseWorker.__init__ does full DB/queue/env setup, so these construct a bare
instance via __new__ and set only the attributes each method under test
needs, rather than going through the real constructor.
"""
import json
import logging
import os
import sys

import pytest

# base_worker.py does unqualified sibling imports (e.g. `from service_config
# import ...`), so it needs workers/ on sys.path directly, same as when it's
# run as a standalone script -- not just importable as workers.base_worker.
_WORKERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workers")
if _WORKERS_DIR not in sys.path:
    sys.path.insert(0, _WORKERS_DIR)

from workers.base_worker import BaseWorker


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def _bare_worker(*, service_host="localhost", service_port=1234, service_name="yolo_v8"):
    worker = BaseWorker.__new__(BaseWorker)
    worker.service_host = service_host
    worker.service_port = service_port
    worker.service_name = service_name
    worker.logger = logging.getLogger("test_base_worker_health")
    return worker


def test_has_coupled_service_true_when_host_and_port_set():
    worker = _bare_worker()
    assert worker.has_coupled_service() is True


def test_has_coupled_service_false_for_db_only_workers():
    worker = _bare_worker(service_host=None, service_port=None)
    assert worker.has_coupled_service() is False


def test_check_service_health_true_when_no_coupled_service():
    worker = _bare_worker(service_host=None, service_port=None)
    assert worker.check_service_health() is True


def test_check_service_health_true_on_200_healthy(monkeypatch):
    worker = _bare_worker()
    monkeypatch.setattr(
        "workers.base_worker.requests.get",
        lambda url, timeout: FakeResponse(200, {"status": "healthy"}),
    )
    assert worker.check_service_health() is True


def test_check_service_health_true_on_200_degraded(monkeypatch):
    """Regression: a 200 'degraded' response (e.g. a service intentionally
    running on CPU instead of GPU to take pressure off shared GPU resources)
    is still able to serve and must not be treated as unhealthy. Caught
    2026-08-20 when this evicted a perfectly working nudenet worker."""
    worker = _bare_worker()
    monkeypatch.setattr(
        "workers.base_worker.requests.get",
        lambda url, timeout: FakeResponse(200, {
            "status": "degraded",
            "warnings": ["DEVICE requested GPU but ONNX Runtime is using CPUExecutionProvider"],
        }),
    )
    assert worker.check_service_health() is True


def test_check_service_health_false_on_503_unhealthy(monkeypatch):
    worker = _bare_worker()
    monkeypatch.setattr(
        "workers.base_worker.requests.get",
        lambda url, timeout: FakeResponse(503, {"status": "unhealthy", "reason": "model error"}),
    )
    assert worker.check_service_health() is False


def test_check_service_health_false_on_exception(monkeypatch):
    worker = _bare_worker()

    def raise_timeout(url, timeout):
        raise TimeoutError("no response")

    monkeypatch.setattr("workers.base_worker.requests.get", raise_timeout)
    assert worker.check_service_health() is False


def test_get_health_url_is_always_the_health_path():
    worker = _bare_worker(service_host="localhost", service_port=7772)
    assert worker.get_health_url() == "http://localhost:7772/health"


def test_shutdown_marker_written_with_expected_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["/home/sd/windmill/workers/yolov8_worker.py"])
    worker = _bare_worker(service_name="yolo_v8")

    worker._write_shutdown_marker("failed post-job health check")

    marker_path = tmp_path / "logs" / "yolov8_worker.shutdown_reason"
    assert marker_path.exists()
    data = json.loads(marker_path.read_text())
    assert data["service"] == "yolo_v8"
    assert data["reason"] == "failed post-job health check"
    assert data["health_url"] == "http://localhost:1234/health"
    assert "shutdown_at" in data


def test_shutdown_unhealthy_writes_marker_and_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["/home/sd/windmill/workers/yolov8_worker.py"])
    worker = _bare_worker(service_name="yolo_v8")

    with pytest.raises(SystemExit) as exc_info:
        worker._shutdown_unhealthy("failed startup health check")

    assert exc_info.value.code == 1
    marker_path = tmp_path / "logs" / "yolov8_worker.shutdown_reason"
    assert marker_path.exists()

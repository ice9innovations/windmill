"""Stable, collision-resistant identity helpers for Windmill workers."""

import os
import re
import socket
import uuid


RESULT_WORKER_ID_MAX_LENGTH = 50


def _component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-") or "unknown"


def create_worker_id(service: str, *, host: str = None, pid: int = None, nonce: str = None) -> str:
    """Return a host-attributable ID that fits results.worker_id varchar(50).

    PID separates concurrent processes on one host and the random suffix makes
    identities unique across rapid restarts and PID reuse.
    """
    host = _component(host or socket.gethostname())
    service = _component(service)
    pid = os.getpid() if pid is None else pid
    nonce = _component(nonce or uuid.uuid4().hex[:8])
    suffix = f"_{host}_{pid}_{nonce}"
    service_budget = RESULT_WORKER_ID_MAX_LENGTH - len("worker_") - len(suffix)
    if service_budget < 1:
        host_budget = max(1, len(host) + service_budget - 1)
        host = host[:host_budget]
        suffix = f"_{host}_{pid}_{nonce}"
        service_budget = 1
    return f"worker_{service[:service_budget]}{suffix}"

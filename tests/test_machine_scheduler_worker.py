from types import SimpleNamespace

from workers.machine_scheduler_worker import DEFAULT_QUEUE_PRIORITY, MachineScheduler, SchedulerSlot


def test_scheduler_slot_orders_workers_by_queue_priority():
    slot = SchedulerSlot.__new__(SchedulerSlot)
    workers = [
        SimpleNamespace(service_name="primary.colors", queue_name="colors"),
        SimpleNamespace(service_name="primary.yolo_v8", queue_name="yolo_v8"),
        SimpleNamespace(service_name="primary.metadata", queue_name="metadata"),
    ]

    ordered = slot._prioritize_workers(workers, ["yolo_v8", "metadata"])

    assert [worker.queue_name for worker in ordered] == ["yolo_v8", "metadata", "colors"]


def test_scheduler_slot_preserves_enabled_order_without_priority():
    slot = SchedulerSlot.__new__(SchedulerSlot)
    workers = [
        SimpleNamespace(service_name="primary.colors", queue_name="colors"),
        SimpleNamespace(service_name="primary.yolo_v8", queue_name="yolo_v8"),
    ]

    ordered = slot._prioritize_workers(workers, [])

    assert ordered == workers


def test_scheduler_slot_default_priority_moves_yolo_ahead_of_enabled_order():
    slot = SchedulerSlot.__new__(SchedulerSlot)
    workers = [
        SimpleNamespace(service_name="primary.colors", queue_name="colors"),
        SimpleNamespace(service_name="primary.face", queue_name="face"),
        SimpleNamespace(service_name="primary.metadata", queue_name="metadata"),
        SimpleNamespace(service_name="primary.nsfw2", queue_name="nsfw2"),
        SimpleNamespace(service_name="primary.nudenet", queue_name="nudenet"),
        SimpleNamespace(service_name="primary.pose", queue_name="pose"),
        SimpleNamespace(service_name="primary.qr", queue_name="qr"),
        SimpleNamespace(service_name="primary.yolo_v8", queue_name="yolo_v8"),
    ]

    ordered = slot._prioritize_workers(workers, DEFAULT_QUEUE_PRIORITY)

    assert [worker.queue_name for worker in ordered] == [
        "yolo_v8",
        "nudenet",
        "nsfw2",
        "face",
        "pose",
        "colors",
        "metadata",
        "qr",
    ]


def test_scheduler_default_priority_prefers_yolo_without_env_override(monkeypatch):
    monkeypatch.delenv("WINDMILL_SCHEDULER_QUEUE_PRIORITY", raising=False)
    scheduler = MachineScheduler.__new__(MachineScheduler)

    assert scheduler._load_queue_priority()[:3] == ["yolo_v8", "nudenet", "nsfw2"]

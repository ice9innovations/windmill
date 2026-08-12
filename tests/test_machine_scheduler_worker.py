from types import SimpleNamespace

from workers.machine_scheduler_worker import SchedulerSlot


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

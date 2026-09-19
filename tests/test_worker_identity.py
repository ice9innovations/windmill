from core.worker_identity import RESULT_WORKER_ID_MAX_LENGTH, create_worker_id


def test_worker_id_is_host_attributable_and_fits_results_column():
    worker_id = create_worker_id(
        "system.postprocessing_orchestrator",
        host="dorothy",
        pid=12345,
        nonce="deadbeef",
    )

    assert len(worker_id) <= RESULT_WORKER_ID_MAX_LENGTH
    assert "dorothy" in worker_id
    assert worker_id.endswith("_12345_deadbeef")


def test_worker_ids_do_not_collide_for_simultaneous_processes():
    first = create_worker_id("primary.yolo_v8", host="orin", pid=100, nonce="aaaaaaaa")
    second = create_worker_id("primary.yolo_v8", host="orin", pid=101, nonce="aaaaaaaa")
    restarted = create_worker_id("primary.yolo_v8", host="orin", pid=100, nonce="bbbbbbbb")

    assert len({first, second, restarted}) == 3

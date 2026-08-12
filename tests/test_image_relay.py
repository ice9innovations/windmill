import builtins
import threading
import time

import pytest

from core.image_store import ImageStoreConfig
from image_relay import RelayConfig, SingleFlightRelay, TTLRUCache


def relay_config(**overrides):
    values = {
        "bind_host": "127.0.0.1",
        "port": 8787,
        "allowed_cidrs": (),
        "cache_ttl_seconds": 1.0,
        "max_cache_bytes": 1024,
        "max_object_bytes": 1024,
        "waiter_timeout_seconds": 2.0,
    }
    values.update(overrides)
    return RelayConfig(**values)


def image_store_config(**overrides):
    values = {
        "mode": "valkey",
        "host": "images.example",
        "port": 6379,
        "use_ssl": True,
        "username": "windmill",
        "password": "secret",
        "ca_certs": "/tmp/ca.pem",
        "image_ttl_seconds": 90,
        "crop_ttl_seconds": 90,
        "socket_connect_timeout_seconds": 3,
        "socket_timeout_seconds": 3,
        "health_check_interval_seconds": 30,
        "keepalive_ping_seconds": 15,
        "relay_url": None,
        "relay_timeout_seconds": 5,
        "relay_fallback_direct": False,
    }
    values.update(overrides)
    return ImageStoreConfig(**values)


class FakeRelay(SingleFlightRelay):
    def __init__(self, payloads=None, errors=None, delay=0.0, start_barrier=None, **kwargs):
        super().__init__(
            cache=TTLRUCache(
                max_bytes=kwargs.get("max_cache_bytes", 1024),
                max_object_bytes=kwargs.get("max_object_bytes", 1024),
                now=kwargs.get("now", time.monotonic),
            ),
            relay_config=relay_config(
                cache_ttl_seconds=kwargs.get("cache_ttl_seconds", 1.0),
                max_cache_bytes=kwargs.get("max_cache_bytes", 1024),
                max_object_bytes=kwargs.get("max_object_bytes", 1024),
            ),
            image_store_config=image_store_config(**kwargs.get("image_store_overrides", {})),
        )
        self.payloads = payloads or {}
        self.errors = errors or {}
        self.delay = delay
        self.start_barrier = start_barrier
        self.calls = []
        self._calls_lock = threading.Lock()

    def _fetch_upstream(self, kind, ref):
        with self._calls_lock:
            self.calls.append((kind, ref, time.monotonic()))
        if self.start_barrier is not None:
            self.start_barrier.wait()
        if self.delay:
            time.sleep(self.delay)
        if ref in self.errors:
            raise self.errors[ref]
        return self.payloads.get(ref, f"payload-{ref}".encode())


def test_concurrent_same_key_requests_share_one_upstream_call():
    relay = FakeRelay(delay=0.1)
    start = threading.Barrier(8)
    results = []

    def worker():
        start.wait()
        results.append(relay.get("image", "wm:image:same"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [b"payload-wm:image:same"] * 8
    assert relay.calls == [("image", "wm:image:same", relay.calls[0][2])]
    metrics = relay.metrics()
    assert metrics["requests"] == 8
    assert metrics["cache_misses"] == 1
    assert metrics["singleflight_waiters"] == 7
    assert metrics["singleflight_coalesced"] == 7
    assert metrics["upstream_fetches"] == 1


def test_different_keys_can_fetch_concurrently():
    upstream_started = threading.Barrier(2, timeout=1)
    relay = FakeRelay(start_barrier=upstream_started)
    results = []

    threads = [
        threading.Thread(target=lambda: results.append(relay.get("image", "wm:image:a"))),
        threading.Thread(target=lambda: results.append(relay.get("image", "wm:image:b"))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [b"payload-wm:image:a", b"payload-wm:image:b"]
    assert sorted(call[1] for call in relay.calls) == ["wm:image:a", "wm:image:b"]


def test_ttl_expiry_fetches_again():
    now = [100.0]
    relay = FakeRelay(now=lambda: now[0], cache_ttl_seconds=1.0)

    assert relay.get("image", "wm:image:a") == b"payload-wm:image:a"
    assert relay.get("image", "wm:image:a") == b"payload-wm:image:a"
    now[0] = 101.1
    assert relay.get("image", "wm:image:a") == b"payload-wm:image:a"

    assert len(relay.calls) == 2


def test_cache_ttl_is_clamped_below_backing_valkey_ttl():
    now = [100.0]
    relay = FakeRelay(
        now=lambda: now[0],
        cache_ttl_seconds=90.0,
        image_store_overrides={"image_ttl_seconds": 90},
    )

    assert relay.get("image", "wm:image:a") == b"payload-wm:image:a"
    now[0] = 171.9
    assert relay.get("image", "wm:image:a") == b"payload-wm:image:a"
    now[0] = 172.1
    assert relay.get("image", "wm:image:a") == b"payload-wm:image:a"

    assert len(relay.calls) == 2


def test_bounded_memory_lru_eviction():
    cache = TTLRUCache(max_bytes=6, max_object_bytes=10, now=lambda: 1.0)
    assert cache.put(("image", "a"), b"aa", 10)
    assert cache.put(("image", "b"), b"bb", 10)
    assert cache.get(("image", "a")) == b"aa"
    assert cache.put(("image", "c"), b"ccc", 10)

    assert cache.get(("image", "a")) == b"aa"
    assert cache.get(("image", "b")) is None
    assert cache.get(("image", "c")) == b"ccc"
    assert cache.current_bytes == 5


def test_maximum_object_size_is_enforced():
    relay = FakeRelay(payloads={"wm:image:big": b"abcd"}, max_object_bytes=3)

    with pytest.raises(ValueError, match="maximum object size"):
        relay.get("image", "wm:image:big")

    assert relay.cache.stats()["entries"] == 0
    assert relay.metrics()["object_too_large"] == 1


def test_upstream_error_propagates_to_all_waiters():
    relay = FakeRelay(errors={"wm:image:bad": RuntimeError("upstream failed")}, delay=0.1)
    start = threading.Barrier(3)
    errors = []

    def worker():
        start.wait()
        with pytest.raises(RuntimeError):
            relay.get("image", "wm:image:bad")
        errors.append(True)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 3
    assert len(relay.calls) == 1


def test_relay_path_does_not_open_files(monkeypatch):
    def fail_open(*args, **kwargs):
        raise AssertionError("relay image path opened a file")

    relay = FakeRelay(payloads={"wm:image:a": b"bytes"})
    monkeypatch.setattr(builtins, "open", fail_open)

    assert relay.get("image", "wm:image:a") == b"bytes"

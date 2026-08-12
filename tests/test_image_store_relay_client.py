import pytest
import requests

from core import image_store
from core.image_store import ImageStoreConfig


def config(**overrides):
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


def test_direct_behavior_unchanged_when_relay_is_unset(monkeypatch):
    calls = []

    def fake_get_bytes(ref, refresh_ttl_s, config, log):
        calls.append((ref, refresh_ttl_s, config.relay_url))
        return b"direct"

    monkeypatch.setattr(image_store, "get_bytes", fake_get_bytes)

    assert image_store.get_image("wm:image:a", config=config()) == b"direct"
    assert calls == [("wm:image:a", 90, None)]


def test_direct_fetch_stats_include_duration_and_bytes(monkeypatch):
    def fake_get_bytes(ref, refresh_ttl_s, config, log):
        return b"direct"

    monkeypatch.setattr(image_store, "get_bytes", fake_get_bytes)

    payload, stats = image_store.get_image_with_stats("wm:image:a", config=config())

    assert payload == b"direct"
    assert stats.transport == "direct_valkey"
    assert stats.bytes == 6
    assert stats.found is True
    assert stats.direct_duration_seconds is not None


def test_relay_fetch_stats_include_relay_timing(monkeypatch):
    def fake_relay(*args, **kwargs):
        return b"relay"

    monkeypatch.setattr(image_store, "_get_bytes_via_relay", fake_relay)

    payload, stats = image_store.get_image_with_stats(
        "wm:image:a",
        config=config(relay_url="http://192.168.0.101:8787"),
    )

    assert payload == b"relay"
    assert stats.transport == "image_relay"
    assert stats.bytes == 5
    assert stats.relay_duration_seconds is not None
    assert stats.direct_duration_seconds is None


def test_relay_client_timeout_uses_optional_direct_fallback(monkeypatch):
    direct_calls = []

    def fake_relay(*args, **kwargs):
        raise requests.Timeout("slow relay")

    def fake_get_bytes(ref, refresh_ttl_s, config, log):
        direct_calls.append((ref, refresh_ttl_s))
        return b"direct-after-timeout"

    monkeypatch.setattr(image_store, "_get_bytes_via_relay", fake_relay)
    monkeypatch.setattr(image_store, "get_bytes", fake_get_bytes)

    result = image_store.get_image(
        "wm:image:a",
        config=config(relay_url="http://192.168.0.101:8787", relay_fallback_direct=True),
    )

    assert result == b"direct-after-timeout"
    assert direct_calls == [("wm:image:a", 90)]


def test_relay_fetch_stats_include_fallback_timing(monkeypatch):
    def fake_relay(*args, **kwargs):
        raise requests.Timeout("slow relay")

    def fake_get_bytes(ref, refresh_ttl_s, config, log):
        return b"direct-after-timeout"

    monkeypatch.setattr(image_store, "_get_bytes_via_relay", fake_relay)
    monkeypatch.setattr(image_store, "get_bytes", fake_get_bytes)

    payload, stats = image_store.get_image_with_stats(
        "wm:image:a",
        config=config(relay_url="http://192.168.0.101:8787", relay_fallback_direct=True),
    )

    assert payload == b"direct-after-timeout"
    assert stats.transport == "image_relay_fallback_direct"
    assert stats.fallback_direct is True
    assert stats.error == "Timeout"
    assert stats.relay_duration_seconds is not None
    assert stats.direct_duration_seconds is not None


def test_relay_client_timeout_raises_when_fallback_disabled(monkeypatch):
    def fake_relay(*args, **kwargs):
        raise requests.Timeout("slow relay")

    monkeypatch.setattr(image_store, "_get_bytes_via_relay", fake_relay)

    with pytest.raises(requests.Timeout):
        image_store.get_image(
            "wm:image:a",
            config=config(relay_url="http://192.168.0.101:8787", relay_fallback_direct=False),
        )

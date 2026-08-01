#!/usr/bin/env python3
"""
RAM-only Windmill image relay.

The relay coalesces concurrent LAN requests for the same Valkey image ref and
caches successful payloads in memory only. It never writes image bytes to disk.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, request

from core.image_store import get_bytes, get_image_store_config

logger = logging.getLogger("image_relay")

_DEFAULT_ALLOWED_CIDRS = "127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,::1/128,fc00::/7,fe80::/10"
_VALKEY_TTL_CACHE_FRACTION = 0.8


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _ref_log_id(ref: str) -> str:
    return hashlib.sha256(ref.encode("utf-8")).hexdigest()[:12]


def _json_log(event: str, **fields) -> None:
    fields["event"] = event
    logger.info(json.dumps(fields, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class RelayConfig:
    bind_host: str
    port: int
    allowed_cidrs: Tuple[ipaddress._BaseNetwork, ...]
    cache_ttl_seconds: float
    max_cache_bytes: int
    max_object_bytes: int
    waiter_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "RelayConfig":
        allowed = os.getenv("IMAGE_RELAY_ALLOWED_CIDRS", _DEFAULT_ALLOWED_CIDRS)
        cidrs = tuple(ipaddress.ip_network(item.strip()) for item in allowed.split(",") if item.strip())
        return cls(
            bind_host=os.getenv("IMAGE_RELAY_BIND_HOST", "127.0.0.1"),
            port=_env_int("IMAGE_RELAY_PORT", 8787),
            allowed_cidrs=cidrs,
            cache_ttl_seconds=_env_float("IMAGE_RELAY_CACHE_TTL_SECONDS", 15.0),
            max_cache_bytes=_env_int("IMAGE_RELAY_MAX_CACHE_BYTES", 512 * 1024 * 1024),
            max_object_bytes=_env_int("IMAGE_RELAY_MAX_OBJECT_BYTES", 64 * 1024 * 1024),
            waiter_timeout_seconds=_env_float("IMAGE_RELAY_WAITER_TIMEOUT_SECONDS", 30.0),
        )

    def validate(self) -> None:
        if self.port <= 0:
            raise ValueError("IMAGE_RELAY_PORT must be positive")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("IMAGE_RELAY_CACHE_TTL_SECONDS must be positive")
        if self.max_cache_bytes <= 0:
            raise ValueError("IMAGE_RELAY_MAX_CACHE_BYTES must be positive")
        if self.max_object_bytes <= 0:
            raise ValueError("IMAGE_RELAY_MAX_OBJECT_BYTES must be positive")
        if self.waiter_timeout_seconds <= 0:
            raise ValueError("IMAGE_RELAY_WAITER_TIMEOUT_SECONDS must be positive")


class TTLRUCache:
    def __init__(
        self,
        *,
        max_bytes: int,
        max_object_bytes: int,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_object_bytes = max_object_bytes
        self._now = now
        self._lock = threading.Lock()
        self._entries: "OrderedDict[Tuple[str, str], Tuple[bytes, int, float]]" = OrderedDict()
        self.current_bytes = 0

    def get(self, key: Tuple[str, str]) -> Optional[bytes]:
        now = self._now()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            payload, size, expires_at = entry
            if expires_at <= now:
                self._delete_locked(key)
                return None
            self._entries.move_to_end(key)
            return payload

    def put(self, key: Tuple[str, str], payload: bytes, ttl_seconds: float) -> bool:
        size = len(payload)
        if size > self.max_object_bytes:
            return False
        now = self._now()
        with self._lock:
            if key in self._entries:
                self._delete_locked(key)
            self._entries[key] = (payload, size, now + ttl_seconds)
            self.current_bytes += size
            self._evict_locked(now)
            return key in self._entries

    def _delete_locked(self, key: Tuple[str, str]) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self.current_bytes -= entry[1]

    def _evict_locked(self, now: float) -> None:
        expired = [key for key, (_, _, expires_at) in self._entries.items() if expires_at <= now]
        for key in expired:
            self._delete_locked(key)
        while self.current_bytes > self.max_bytes and self._entries:
            key, _ = next(iter(self._entries.items()))
            self._delete_locked(key)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "current_bytes": self.current_bytes,
                "max_bytes": self.max_bytes,
                "max_object_bytes": self.max_object_bytes,
            }


@dataclass
class InFlightFetch:
    event: threading.Event
    payload: Optional[bytes] = None
    error: Optional[BaseException] = None


class SingleFlightRelay:
    def __init__(self, *, cache: TTLRUCache, relay_config: RelayConfig, image_store_config=None) -> None:
        self.cache = cache
        self.relay_config = relay_config
        self.image_store_config = image_store_config or get_image_store_config()
        self._lock = threading.RLock()
        self._inflight: Dict[Tuple[str, str], InFlightFetch] = {}
        self._metrics = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "singleflight_waiters": 0,
            "singleflight_coalesced": 0,
            "upstream_fetches": 0,
            "upstream_errors": 0,
            "object_too_large": 0,
            "not_found": 0,
        }

    def get(self, kind: str, ref: str) -> Optional[bytes]:
        self._inc("requests")
        key = (kind, ref)
        payload = self.cache.get(key)
        if payload is not None:
            self._inc("cache_hits")
            _json_log(
                "cache_hit",
                kind=kind,
                ref_id=_ref_log_id(ref),
                bytes=len(payload),
            )
            return payload

        with self._lock:
            flight = self._inflight.get(key)
            if flight is None:
                flight = InFlightFetch(event=threading.Event())
                self._inflight[key] = flight
                leader = True
                self._inc("cache_misses")
                _json_log("cache_miss_leader", kind=kind, ref_id=_ref_log_id(ref))
            else:
                leader = False
                self._inc("singleflight_waiters")
                self._inc("singleflight_coalesced")
                _json_log("singleflight_waiter", kind=kind, ref_id=_ref_log_id(ref))

        if leader:
            self._fetch_as_leader(kind, ref, key, flight)
        else:
            if not flight.event.wait(self.relay_config.waiter_timeout_seconds):
                raise TimeoutError("Timed out waiting for in-flight image fetch")

        if leader:
            with self._lock:
                self._inflight.pop(key, None)
        if flight.error is not None:
            raise flight.error
        return flight.payload

    def _fetch_as_leader(self, kind: str, ref: str, key: Tuple[str, str], flight: InFlightFetch) -> None:
        payload = None
        error = None
        start = time.perf_counter()
        try:
            self._inc("upstream_fetches")
            payload = self._fetch_upstream(kind, ref)
            duration = time.perf_counter() - start
            payload_size = len(payload) if payload is not None else 0
            mbps = (payload_size * 8 / 1_000_000 / duration) if duration > 0 else 0
            _json_log(
                "upstream_fetch",
                kind=kind,
                ref_id=_ref_log_id(ref),
                duration_seconds=round(duration, 6),
                bytes=payload_size,
                effective_mbps=round(mbps, 3),
                found=payload is not None,
            )
            if payload is None:
                self._inc("not_found")
            elif len(payload) > self.relay_config.max_object_bytes:
                self._inc("object_too_large")
                error = ValueError("Image exceeds relay maximum object size")
            else:
                ttl = self._effective_cache_ttl(kind)
                self.cache.put(key, payload, ttl)
        except Exception as exc:
            self._inc("upstream_errors")
            _json_log(
                "upstream_error",
                kind=kind,
                ref_id=_ref_log_id(ref),
                error=exc.__class__.__name__,
            )
            error = exc
        finally:
            flight.payload = payload if error is None else None
            flight.error = error
            flight.event.set()

    def _fetch_upstream(self, kind: str, ref: str) -> Optional[bytes]:
        if kind == "image":
            return get_bytes(
                ref,
                refresh_ttl_s=self.image_store_config.image_ttl_seconds,
                config=self.image_store_config,
                log=logger,
            )
        if kind == "crop":
            return get_bytes(
                ref,
                refresh_ttl_s=self.image_store_config.crop_ttl_seconds,
                config=self.image_store_config,
                log=logger,
            )
        raise ValueError(f"Unsupported relay kind: {kind}")

    def _effective_cache_ttl(self, kind: str) -> float:
        backing_ttl = (
            self.image_store_config.image_ttl_seconds
            if kind == "image"
            else self.image_store_config.crop_ttl_seconds
        )
        return min(
            self.relay_config.cache_ttl_seconds,
            max(0.001, backing_ttl * _VALKEY_TTL_CACHE_FRACTION),
        )

    def _inc(self, metric: str) -> None:
        with self._lock:
            self._metrics[metric] = self._metrics.get(metric, 0) + 1

    def metrics(self) -> Dict[str, int]:
        with self._lock:
            data = dict(self._metrics)
        data.update(self.cache.stats())
        return data


def _client_allowed(remote_addr: str, allowed_cidrs: Iterable[ipaddress._BaseNetwork]) -> bool:
    try:
        client_ip = ipaddress.ip_address(remote_addr)
    except ValueError:
        return False
    return any(client_ip in network for network in allowed_cidrs)


def create_app(relay_config: Optional[RelayConfig] = None, relay: Optional[SingleFlightRelay] = None) -> Flask:
    relay_config = relay_config or RelayConfig.from_env()
    relay_config.validate()
    cache = TTLRUCache(
        max_bytes=relay_config.max_cache_bytes,
        max_object_bytes=relay_config.max_object_bytes,
    )
    relay = relay or SingleFlightRelay(cache=cache, relay_config=relay_config)
    app = Flask(__name__)
    app.config["relay"] = relay
    app.config["relay_config"] = relay_config
    app.config["MAX_CONTENT_LENGTH"] = 0

    @app.before_request
    def enforce_lan_access():
        if not _client_allowed(request.remote_addr or "", relay_config.allowed_cidrs):
            abort(403)

    @app.get("/healthz")
    def healthz():
        return jsonify({
            "ok": True,
            "hostname": socket.gethostname(),
            "cache": relay.cache.stats(),
        })

    @app.get("/metrics")
    def metrics():
        return jsonify(relay.metrics())

    @app.get("/image/<path:ref>")
    def get_relay_image(ref: str):
        kind = request.args.get("kind", "image")
        started_at = time.perf_counter()
        attribution = {
            "hostname": request.headers.get("X-Windmill-Hostname", "-"),
            "service": request.headers.get("X-Windmill-Service", "-"),
            "worker_id": request.headers.get("X-Windmill-Worker-Id", "-"),
            "image_id": request.headers.get("X-Windmill-Image-Id", "-"),
            "trace_id": request.headers.get("X-Windmill-Trace-Id", "-"),
        }
        try:
            payload = relay.get(kind, ref)
            status = 200 if payload is not None else 404
            if payload is None:
                response = Response(status=404)
            else:
                response = Response(payload, mimetype="application/octet-stream")
                response.headers["Cache-Control"] = "no-store"
        except ValueError as exc:
            status = 413 if "maximum object size" in str(exc) else 400
            response = Response(str(exc), status=status, mimetype="text/plain")
        except Exception as exc:
            status = 502
            response = Response(exc.__class__.__name__, status=status, mimetype="text/plain")
        duration = time.perf_counter() - started_at
        byte_count = len(payload) if "payload" in locals() and payload is not None else 0
        mbps = (byte_count * 8 / 1_000_000 / duration) if duration > 0 else 0
        _json_log(
            "relay_request",
            kind=kind,
            ref_id=_ref_log_id(ref),
            status=status,
            duration_seconds=round(duration, 6),
            bytes=byte_count,
            effective_mbps=round(mbps, 3),
            **attribution,
        )
        return response

    return app


def main() -> None:
    load_dotenv(os.getenv("WINDMILL_ENV_FILE", ".env"), override=False)
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    relay_config = RelayConfig.from_env()
    relay_config.validate()
    logger.info(
        "Starting image relay host=%s port=%d allowed_cidrs=%s max_cache_bytes=%d max_object_bytes=%d",
        relay_config.bind_host,
        relay_config.port,
        ",".join(str(network) for network in relay_config.allowed_cidrs),
        relay_config.max_cache_bytes,
        relay_config.max_object_bytes,
    )
    create_app(relay_config=relay_config).run(
        host=relay_config.bind_host,
        port=relay_config.port,
        threaded=True,
    )


if __name__ == "__main__":
    main()

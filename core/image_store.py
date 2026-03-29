"""
Ephemeral image storage helpers.

Supports the current inline transport mode and an opt-in Valkey-backed mode
that stores image bytes out-of-band and passes opaque refs through RabbitMQ.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import redis

_DEFAULT_CA_CERTS = "/etc/ssl/certs/ca-certificates.crt"
_DEFAULT_IMAGE_TTL_SECONDS = 90
_DEFAULT_CROP_TTL_SECONDS = 90
_DEFAULT_KEY_PREFIX = "wm"

_client = None
_client_lock = Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


@dataclass(frozen=True)
class ImageStoreConfig:
    mode: str
    host: Optional[str]
    port: int
    use_ssl: bool
    username: Optional[str]
    password: Optional[str]
    ca_certs: Optional[str]
    image_ttl_seconds: int
    crop_ttl_seconds: int
    key_prefix: str = _DEFAULT_KEY_PREFIX

    @property
    def enabled(self) -> bool:
        return self.mode == "valkey"

    @classmethod
    def from_env(cls) -> "ImageStoreConfig":
        mode = (os.getenv("IMAGE_STORE_MODE", "inline") or "inline").strip().lower()
        return cls(
            mode=mode,
            host=os.getenv("VALKEY_HOST"),
            port=_env_int("VALKEY_PORT", 6379),
            use_ssl=_env_bool("VALKEY_SSL", True),
            username=os.getenv("VALKEY_USERNAME") or None,
            password=os.getenv("VALKEY_PASSWORD") or None,
            ca_certs=os.getenv("VALKEY_CA_CERTS", _DEFAULT_CA_CERTS) or None,
            image_ttl_seconds=_env_int("VALKEY_IMAGE_TTL_SECONDS", _DEFAULT_IMAGE_TTL_SECONDS),
            crop_ttl_seconds=_env_int("VALKEY_CROP_TTL_SECONDS", _DEFAULT_CROP_TTL_SECONDS),
        )

    def validate(self) -> None:
        if self.mode not in ("inline", "valkey"):
            raise ValueError(f"Unsupported IMAGE_STORE_MODE '{self.mode}'")
        if not self.enabled:
            return
        if not self.host:
            raise ValueError("VALKEY_HOST is required when IMAGE_STORE_MODE=valkey")
        if self.port <= 0:
            raise ValueError("VALKEY_PORT must be a positive integer")
        if self.image_ttl_seconds <= 0:
            raise ValueError("VALKEY_IMAGE_TTL_SECONDS must be a positive integer")
        if self.crop_ttl_seconds <= 0:
            raise ValueError("VALKEY_CROP_TTL_SECONDS must be a positive integer")


def get_image_store_config() -> ImageStoreConfig:
    config = ImageStoreConfig.from_env()
    config.validate()
    return config


def is_valkey_image_store_enabled() -> bool:
    return get_image_store_config().enabled


def get_client(config: Optional[ImageStoreConfig] = None):
    global _client
    config = config or get_image_store_config()
    if not config.enabled:
        raise RuntimeError("Valkey image store requested while IMAGE_STORE_MODE is not 'valkey'")

    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client
        kwargs = {
            "host": config.host,
            "port": config.port,
            "username": config.username,
            "password": config.password,
            "ssl": config.use_ssl,
            "decode_responses": False,
        }
        if config.use_ssl and config.ca_certs:
            kwargs["ssl_ca_certs"] = config.ca_certs
        _client = redis.Redis(**kwargs)
        return _client


def ping(config: Optional[ImageStoreConfig] = None) -> bool:
    return bool(get_client(config).ping())


def _make_ref(kind: str, key_prefix: str = _DEFAULT_KEY_PREFIX) -> str:
    return f"{key_prefix}:{kind}:{uuid.uuid4().hex}"


def put_bytes(ref: str, payload: bytes, ttl_s: int, config: Optional[ImageStoreConfig] = None) -> str:
    get_client(config).set(ref, payload, ex=ttl_s)
    return ref


def put_image(image_bytes: bytes, ttl_s: Optional[int] = None, config: Optional[ImageStoreConfig] = None) -> str:
    config = config or get_image_store_config()
    ttl_s = ttl_s or config.image_ttl_seconds
    ref = _make_ref("image", key_prefix=config.key_prefix)
    return put_bytes(ref, image_bytes, ttl_s, config=config)


def put_crop(crop_bytes: bytes, ttl_s: Optional[int] = None, config: Optional[ImageStoreConfig] = None) -> str:
    config = config or get_image_store_config()
    ttl_s = ttl_s or config.crop_ttl_seconds
    ref = _make_ref("crop", key_prefix=config.key_prefix)
    return put_bytes(ref, crop_bytes, ttl_s, config=config)


def get_bytes(ref: str, refresh_ttl_s: Optional[int] = None, config: Optional[ImageStoreConfig] = None) -> Optional[bytes]:
    client = get_client(config)
    if refresh_ttl_s:
        return client.getex(ref, ex=refresh_ttl_s)
    return client.get(ref)


def get_image(ref: str, refresh_ttl_s: Optional[int] = None, config: Optional[ImageStoreConfig] = None) -> Optional[bytes]:
    config = config or get_image_store_config()
    refresh_ttl_s = refresh_ttl_s or config.image_ttl_seconds
    return get_bytes(ref, refresh_ttl_s=refresh_ttl_s, config=config)


def get_crop(ref: str, refresh_ttl_s: Optional[int] = None, config: Optional[ImageStoreConfig] = None) -> Optional[bytes]:
    config = config or get_image_store_config()
    refresh_ttl_s = refresh_ttl_s or config.crop_ttl_seconds
    return get_bytes(ref, refresh_ttl_s=refresh_ttl_s, config=config)


def touch(ref: str, ttl_s: int, config: Optional[ImageStoreConfig] = None) -> bool:
    return bool(get_client(config).expire(ref, ttl_s))


def delete(ref: str, config: Optional[ImageStoreConfig] = None) -> int:
    return int(get_client(config).delete(ref))


def ttl(ref: str, config: Optional[ImageStoreConfig] = None) -> int:
    return int(get_client(config).ttl(ref))

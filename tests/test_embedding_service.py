import json
import os

import pytest

from core.dispatch import resolve_services
from workers.embedding_worker import EmbeddingWorker
from workers.service_config import ServiceConfig


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "service_config.yaml")


def test_embedding_tier_is_basic_plus_embedding():
    config = ServiceConfig(CONFIG_PATH)

    assert set(config.get_services_by_tier("embedding")) == (
        set(config.get_services_by_tier("basic")) | {"primary.embedding"}
    )
    assert set(resolve_services("embedding", config)) == (
        set(resolve_services("basic", config)) | {"embedding"}
    )
    assert config.get_queue_name("primary.embedding") == "embedding"
    assert config.get_service_config("primary.embedding")["endpoint"] == "/embed/image"
    assert config.is_available_for_tier("system.content_analysis", "embedding")


def test_validate_embedding_requires_768_finite_numbers():
    embedding = [0.0] * 768
    assert EmbeddingWorker.validate_embedding({"image_embedding": embedding}) is embedding

    with pytest.raises(ValueError):
        EmbeddingWorker.validate_embedding({"image_embedding": [0.0] * 767})
    with pytest.raises(ValueError):
        EmbeddingWorker.validate_embedding({"image_embedding": [0.0] * 767 + [float("nan")]})


def test_embedding_worker_persists_vector_once():
    class Cursor:
        def __init__(self):
            self.sql = None
            self.params = None
            self.closed = False

        def execute(self, sql, params):
            self.sql = sql
            self.params = params

        def close(self):
            self.closed = True

    cursor = Cursor()
    worker = object.__new__(EmbeddingWorker)
    worker.db_conn = type("Connection", (), {"cursor": lambda self: cursor})()
    embedding = [0.0] * 768

    worker.after_result_stored(42, {"image_embedding": embedding}, {})

    assert "image_clip_embedding IS NULL" in cursor.sql
    assert json.loads(cursor.params[0]) == embedding
    assert cursor.params[1] == 42
    assert cursor.closed

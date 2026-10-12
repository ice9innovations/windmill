#!/usr/bin/env python3
"""Generate and persist normalized CLIP image embeddings."""

import json
import math
import os
import sys

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker


class EmbeddingWorker(BaseWorker):
    """Worker for Animal Farm's CLIP ``/embed/image`` endpoint."""

    EMBEDDING_DIMENSIONS = 768

    def __init__(self):
        super().__init__('primary.embedding')

    @classmethod
    def validate_embedding(cls, result):
        embedding = result.get('image_embedding') if isinstance(result, dict) else None
        if not isinstance(embedding, list) or len(embedding) != cls.EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"CLIP response must contain a {cls.EMBEDDING_DIMENSIONS}-dimensional "
                "image_embedding"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in embedding
        ):
            raise ValueError("CLIP image_embedding contains a non-finite or non-numeric value")
        return embedding

    def after_result_stored(self, image_id, result, message):
        """Store the vector in its indexed pgvector column.

        The NULL guard keeps retries and duplicate submissions idempotent.
        """
        embedding = self.validate_embedding(result)
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE images
                   SET image_clip_embedding = %s::vector
                 WHERE image_id = %s
                   AND image_clip_embedding IS NULL
                """,
                (json.dumps(embedding), image_id),
            )
        finally:
            cursor.close()


if __name__ == "__main__":
    EmbeddingWorker().start()

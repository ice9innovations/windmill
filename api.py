#!/usr/bin/env python3
"""
Windmill API - Single image submission and result retrieval
Thin Flask gateway to the existing RabbitMQ processing pipeline
"""
import os
import sys
import uuid
import json
import base64
import io
import logging

import pika
import psycopg2
import psycopg2.extras
import requests as http_requests
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Add workers directory to path for service_config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'workers'))
from service_config import get_service_config

load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _require_env(key):
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable {key} not set")
    return value

QUEUE_HOST = _require_env('QUEUE_HOST')
QUEUE_USER = _require_env('QUEUE_USER')
QUEUE_PASSWORD = _require_env('QUEUE_PASSWORD')
DB_HOST = _require_env('DB_HOST')
DB_NAME = _require_env('DB_NAME')
DB_USER = _require_env('DB_USER')
DB_PASSWORD = _require_env('DB_PASSWORD')

config = get_service_config('service_config.yaml')

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_db_conn = None
_rabbit_conn = None
_rabbit_channel = None


def get_db():
    """Get or create a database connection."""
    global _db_conn
    try:
        if _db_conn is None or _db_conn.closed:
            _db_conn = psycopg2.connect(
                host=DB_HOST, database=DB_NAME,
                user=DB_USER, password=DB_PASSWORD,
            )
            _db_conn.autocommit = True
            app.logger.info("Connected to PostgreSQL at %s", DB_HOST)
        # Verify connection is alive
        _db_conn.cursor().execute("SELECT 1")
        return _db_conn
    except Exception:
        # Connection stale, reconnect
        try:
            if _db_conn:
                _db_conn.close()
        except Exception:
            pass
        _db_conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
        )
        _db_conn.autocommit = True
        app.logger.info("Reconnected to PostgreSQL at %s", DB_HOST)
        return _db_conn


def get_channel():
    """Get or create a RabbitMQ channel."""
    global _rabbit_conn, _rabbit_channel
    try:
        if _rabbit_conn is None or _rabbit_conn.is_closed:
            raise Exception("need new connection")
        if _rabbit_channel is None or _rabbit_channel.is_closed:
            raise Exception("need new channel")
        return _rabbit_channel
    except Exception:
        credentials = pika.PlainCredentials(QUEUE_USER, QUEUE_PASSWORD)
        _rabbit_conn = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=QUEUE_HOST,
                credentials=credentials,
                heartbeat=60,
                blocked_connection_timeout=300,
                connection_attempts=10,
                retry_delay=5,
                socket_timeout=10,
            )
        )
        _rabbit_channel = _rabbit_conn.channel()
        app.logger.info("Connected to RabbitMQ at %s", QUEUE_HOST)
        return _rabbit_channel


def declare_queue(channel, queue_name):
    """Declare a queue with DLQ, matching worker convention."""
    dlq_name = f"{queue_name}.dlq"
    channel.queue_declare(queue=dlq_name, durable=True)
    args = {
        'x-dead-letter-exchange': '',
        'x-dead-letter-routing-key': dlq_name,
    }
    ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
    if ttl_env and ttl_env.isdigit() and int(ttl_env) > 0:
        args['x-message-ttl'] = int(ttl_env)
    channel.queue_declare(queue=queue_name, durable=True, arguments=args)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def extract_image_dimensions(image_bytes):
    """Extract width and height from image bytes using PIL."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.width, image.height
    except Exception:
        return -1, -1


def resolve_services(services_param):
    """Resolve a comma-separated service list (or None for all primary)."""
    primary = config.get_services_by_category('primary')
    available = sorted(name.split('.', 1)[1] for name in primary.keys())

    if not services_param:
        return available, None

    requested = [s.strip() for s in services_param.split(',')]
    invalid = [s for s in requested if s not in available]
    if invalid:
        return None, f"Unknown services: {', '.join(invalid)}. Available: {', '.join(available)}"
    return requested, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/analyze', methods=['POST'])
def analyze():
    """Submit a single image for analysis.

    Accepts either:
      - multipart/form-data with an 'image' file field
      - JSON body with 'image_url'

    Optional params (form field or JSON key):
      - services: comma-separated service list (default: all primary)
      - image_group: group tag (default: 'api')
    """
    image_bytes = None
    image_filename = None

    # --- resolve image bytes ---
    if request.content_type and 'multipart/form-data' in request.content_type:
        file = request.files.get('image')
        if not file:
            return jsonify({"error": "No 'image' file in request"}), 400
        image_bytes = file.read()
        image_filename = file.filename or 'upload.jpg'
        services_param = request.form.get('services')
        image_group = request.form.get('image_group', 'api')
    else:
        data = request.get_json(silent=True) or {}
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({"error": "Provide an 'image' file upload or JSON with 'image_url'"}), 400
        try:
            resp = http_requests.get(image_url, timeout=15)
            resp.raise_for_status()
            image_bytes = resp.content
        except Exception as e:
            return jsonify({"error": f"Failed to fetch image from URL: {e}"}), 400
        image_filename = image_url.rsplit('/', 1)[-1] or 'url_image.jpg'
        services_param = data.get('services')
        image_group = data.get('image_group', 'api')

    if not image_bytes:
        return jsonify({"error": "Empty image data"}), 400

    # --- validate services ---
    service_names, err = resolve_services(services_param)
    if err:
        return jsonify({"error": err}), 400

    # --- extract dimensions ---
    width, height = extract_image_dimensions(image_bytes)

    # --- insert into images table ---
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(
            """INSERT INTO images (image_filename, image_group, services_submitted)
               VALUES (%s, %s, %s)
               RETURNING image_id""",
            (image_filename, image_group, service_names),
        )
        image_id = cur.fetchone()[0]
    except Exception as e:
        app.logger.error("Database insert failed: %s", e)
        return jsonify({"error": "Failed to register image in database"}), 500

    # --- publish to queues ---
    trace_id = str(uuid.uuid4())
    b64_data = base64.b64encode(image_bytes).decode('utf-8')

    try:
        channel = get_channel()
        for service_name in service_names:
            queue_name = config.get_queue_name(f'primary.{service_name}')
            declare_queue(channel, queue_name)

            message = {
                "image_id": image_id,
                "image_filename": image_filename,
                "image_data": b64_data,
                "image_width": width,
                "image_height": height,
                "submitted_at": datetime.now().isoformat(),
                "trace_id": trace_id,
                "service_name": service_name,
                "queue_name": queue_name,
            }

            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2),
            )
    except Exception as e:
        app.logger.error("Queue publish failed: %s", e)
        return jsonify({
            "error": "Failed to publish to processing queues",
            "image_id": image_id,
        }), 500

    return jsonify({
        "image_id": image_id,
        "trace_id": trace_id,
        "services_submitted": service_names,
    }), 202


def _fetch_results(cur, image_id):
    """Fetch all results data for an image. Shared by status and results endpoints."""
    # Service results
    cur.execute(
        """SELECT service, data, status, processing_time, result_created
           FROM results WHERE image_id = %s AND status = 'success'
           ORDER BY result_created""",
        (image_id,),
    )
    service_results = {}
    for r in cur.fetchall():
        service_results[r['service']] = {
            "data": r['data'],
            "processing_time": r['processing_time'],
            "result_created": r['result_created'].isoformat() if r['result_created'] else None,
        }

    # Harmonized boxes
    cur.execute(
        "SELECT merged_id, merged_data, status, created FROM merged_boxes WHERE image_id = %s",
        (image_id,),
    )
    merged_boxes = []
    for r in cur.fetchall():
        row_dict = dict(r)
        if row_dict.get('created'):
            row_dict['created'] = row_dict['created'].isoformat()
        merged_boxes.append(row_dict)

    # Consensus
    cur.execute(
        """SELECT consensus_data, processing_time, consensus_created
           FROM consensus WHERE image_id = %s
           ORDER BY consensus_created DESC LIMIT 1""",
        (image_id,),
    )
    consensus_row = cur.fetchone()
    consensus = None
    if consensus_row:
        consensus = {
            "consensus_data": consensus_row['consensus_data'],
            "processing_time": consensus_row['processing_time'],
            "consensus_created": consensus_row['consensus_created'].isoformat() if consensus_row['consensus_created'] else None,
        }

    # Content analysis
    cur.execute(
        """SELECT scene_type, intimacy_level, activities_detected, people_count,
                  gender_breakdown, anatomy_exposed, spatial_relationships,
                  person_attributions, semantic_validation, full_analysis, created
           FROM content_analysis WHERE image_id = %s""",
        (image_id,),
    )
    content_row = cur.fetchone()
    content_analysis = dict(content_row) if content_row else None
    if content_analysis and content_analysis.get('created'):
        content_analysis['created'] = content_analysis['created'].isoformat()

    # Postprocessing
    cur.execute(
        """SELECT service, merged_box_id, data, processing_time
           FROM postprocessing WHERE image_id = %s AND status = 'success'""",
        (image_id,),
    )
    postprocessing = [dict(r) for r in cur.fetchall()]

    return {
        "service_results": service_results,
        "merged_boxes": merged_boxes,
        "consensus": consensus,
        "content_analysis": content_analysis,
        "postprocessing": postprocessing,
    }


@app.route('/status/<int:image_id>', methods=['GET'])
def status(image_id):
    """Check processing status and progressive results for an image."""
    try:
        db = get_db()
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Image metadata
        cur.execute(
            """SELECT image_filename, image_group, services_submitted, image_created
               FROM images WHERE image_id = %s""",
            (image_id,),
        )
        image_row = cur.fetchone()
        if not image_row:
            return jsonify({"error": "Image not found"}), 404

        services_submitted = image_row['services_submitted'] or []

        # Per-service completion status
        cur.execute(
            """SELECT service, status, result_created, processing_time
               FROM results WHERE image_id = %s
               ORDER BY result_created""",
            (image_id,),
        )
        completed = {}
        for r in cur.fetchall():
            completed[r['service']] = {
                "status": r['status'],
                "result_created": r['result_created'].isoformat() if r['result_created'] else None,
                "processing_time": r['processing_time'],
            }

        services_pending = [s for s in services_submitted if s not in completed]
        total = len(services_submitted) if services_submitted else 0
        done = len(completed)

        # Full results
        results_data = _fetch_results(cur, image_id)

        return jsonify({
            "image_id": image_id,
            "image_filename": image_row['image_filename'],
            "image_group": image_row['image_group'],
            "image_created": image_row['image_created'].isoformat() if image_row['image_created'] else None,
            "services_submitted": services_submitted,
            "services_completed": completed,
            "services_pending": services_pending,
            "progress": f"{done}/{total}",
            "is_complete": done == total and total > 0,
            "harmony_complete": len(results_data['merged_boxes']) > 0,
            "consensus_complete": results_data['consensus'] is not None,
            "content_analysis_complete": results_data['content_analysis'] is not None,
            **results_data,
        })

    except Exception as e:
        app.logger.error("Status query failed: %s", e)
        return jsonify({"error": "Failed to query status"}), 500


@app.route('/results/<int:image_id>', methods=['GET'])
def results(image_id):
    """Get full analysis results for an image."""
    try:
        db = get_db()
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Verify image exists
        cur.execute(
            """SELECT image_filename, image_group, services_submitted, image_created
               FROM images WHERE image_id = %s""",
            (image_id,),
        )
        image_row = cur.fetchone()
        if not image_row:
            return jsonify({"error": "Image not found"}), 404

        results_data = _fetch_results(cur, image_id)

        return jsonify({
            "image_id": image_id,
            "image_filename": image_row['image_filename'],
            "image_group": image_row['image_group'],
            "image_created": image_row['image_created'].isoformat() if image_row['image_created'] else None,
            "services_submitted": image_row['services_submitted'] or [],
            **results_data,
        })

    except Exception as e:
        app.logger.error("Results query failed: %s", e)
        return jsonify({"error": "Failed to query results"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 9999))
    debug = os.getenv('API_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

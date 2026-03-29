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
import logging

import ssl
import pika
import psycopg2
import psycopg2.extras
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Add workers and core directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'workers'))
sys.path.insert(0, os.path.dirname(__file__))
from service_config import get_service_config
from core.image import validate_and_normalize_image
from core.image_store import is_valkey_image_store_enabled, put_image
from core.dispatch import resolve_services
from core.results import fetch_results
from core.workflow import get_workflow_definition

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

QUEUE_HOST     = _require_env('QUEUE_HOST')
QUEUE_PORT     = int(os.getenv('QUEUE_PORT', '5672'))
QUEUE_SSL      = os.getenv('QUEUE_SSL', '').lower() in ('true', '1', 'yes')
QUEUE_USER     = _require_env('QUEUE_USER')
QUEUE_PASSWORD = _require_env('QUEUE_PASSWORD')
DB_HOST        = _require_env('DB_HOST')
DB_NAME        = _require_env('DB_NAME')
DB_USER        = _require_env('DB_USER')
DB_PASSWORD    = _require_env('DB_PASSWORD')
DB_SSLMODE     = os.getenv('DB_SSLMODE')

# 16 MB hard cap — reject oversized payloads at the WSGI layer.
# MAX_FORM_MEMORY_SIZE matches so Werkzeug never spools uploads to /tmp.
_MAX_UPLOAD_BYTES = 16 * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = _MAX_UPLOAD_BYTES
app.config['MAX_FORM_MEMORY_SIZE'] = _MAX_UPLOAD_BYTES

SERVICE_CONFIG_PATH = os.getenv('SERVICE_CONFIG_PATH', 'service_config.yaml')
config = get_service_config(SERVICE_CONFIG_PATH)

# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options']        = 'DENY'
    response.headers['Referrer-Policy']        = 'no-referrer'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    # API responses must not be cached — results may be sensitive.
    response.headers['Cache-Control'] = 'no-store'
    return response

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_db_conn       = None
_rabbit_conn   = None
_rabbit_channel = None


def _db_connect_kwargs():
    kwargs = dict(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    if DB_SSLMODE:
        kwargs['sslmode'] = DB_SSLMODE
    return kwargs


def get_db():
    """Get or create a database connection."""
    global _db_conn
    try:
        if _db_conn is None or _db_conn.closed:
            _db_conn = psycopg2.connect(**_db_connect_kwargs())
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
        _db_conn = psycopg2.connect(**_db_connect_kwargs())
        _db_conn.autocommit = True
        app.logger.info("Reconnected to PostgreSQL at %s", DB_HOST)
        return _db_conn


def _new_rabbit_channel():
    """Open a fresh RabbitMQ connection and channel."""
    global _rabbit_conn, _rabbit_channel
    credentials = pika.PlainCredentials(QUEUE_USER, QUEUE_PASSWORD)
    kwargs = dict(
        host=QUEUE_HOST,
        port=QUEUE_PORT,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=300,
        connection_attempts=10,
        retry_delay=5,
        socket_timeout=10,
    )
    if QUEUE_SSL:
        ssl_context = ssl.create_default_context()
        kwargs['ssl_options'] = pika.SSLOptions(ssl_context, QUEUE_HOST)
    _rabbit_conn = pika.BlockingConnection(
        pika.ConnectionParameters(**kwargs)
    )
    _rabbit_channel = _rabbit_conn.channel()
    app.logger.info("Connected to RabbitMQ at %s:%s%s", QUEUE_HOST, QUEUE_PORT,
                    " (TLS)" if QUEUE_SSL else "")
    return _rabbit_channel


def get_channel():
    """Get or create a RabbitMQ channel.

    is_closed is unreliable for detecting a TCP reset — the broker can drop
    the connection while pika still thinks it is open. Always returns a
    channel; callers that catch a connection error should call
    _new_rabbit_channel() directly to force a fresh connection.
    """
    global _rabbit_conn, _rabbit_channel
    if _rabbit_conn is None or _rabbit_conn.is_closed:
        return _new_rabbit_channel()
    if _rabbit_channel is None or _rabbit_channel.is_closed:
        return _new_rabbit_channel()
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
# Routes
# ---------------------------------------------------------------------------

@app.route('/workflow', methods=['GET'])
def workflow():
    """Expose the machine-readable Windmill workflow contract."""
    return jsonify(get_workflow_definition())

@app.route('/analyze', methods=['POST'])
def analyze():
    """Submit a single image for analysis.

    Accepts multipart/form-data with a 'file' field containing an image.
    URL-based submission is not supported — send the image bytes directly.

    Optional form fields:
      - image_group:  group tag for the image (default: 'api')
      - tier:         customer tier — free, basic, premium, batch (default: 'free')
    """
    if not request.content_type or 'multipart/form-data' not in request.content_type:
        return jsonify({"error": "Request must be multipart/form-data with a 'file' field"}), 400

    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file in request"}), 400

    if not file.content_type or not file.content_type.startswith('image/'):
        return jsonify({"error": "File must be an image"}), 400

    image_bytes    = file.read()
    image_filename = file.filename or 'upload.jpg'
    image_group    = request.form.get('image_group', 'api')
    tier           = request.form.get('tier', 'free')

    valid_tiers = config.get_valid_tiers()
    if tier not in valid_tiers:
        return jsonify({"error": f"Invalid tier '{tier}'. Valid tiers: {', '.join(sorted(valid_tiers))}"}), 400

    if not image_bytes:
        return jsonify({"error": "Empty image data"}), 400

    # Validate and normalize — entirely in memory, never touches disk
    try:
        (
            image_bytes,
            original_width,
            original_height,
            normalized_width,
            normalized_height,
        ) = validate_and_normalize_image(image_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    service_names = resolve_services(tier, config)

    # Register image metadata and authoritative primary dispatch rows together.
    try:
        db = get_db()
        previous_autocommit = db.autocommit
        db.autocommit = False
        try:
            cur = db.cursor()
            cur.execute(
                """INSERT INTO images (
                       image_filename,
                       image_group,
                       services_submitted,
                       tier,
                       original_image_width,
                       original_image_height,
                       normalized_image_width,
                       normalized_image_height
                   )
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING image_id""",
                (
                    image_filename,
                    image_group,
                    service_names,
                    tier,
                    original_width,
                    original_height,
                    normalized_width,
                    normalized_height,
                ),
            )
            image_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO service_dispatch (image_id, service) SELECT %s, unnest(%s::text[])",
                (image_id, service_names),
            )
            db.commit()
            cur.close()
        except Exception:
            db.rollback()
            raise
        finally:
            db.autocommit = previous_autocommit
    except Exception as e:
        app.logger.error("Submission registration failed: %s", e)
        return jsonify({"error": "Failed to register image in database"}), 500

    # Store image bytes in the configured transport backend before publishing.
    trace_id = str(uuid.uuid4())
    try:
        if is_valkey_image_store_enabled():
            image_transport = {"image_ref": put_image(image_bytes)}
        else:
            image_transport = {"image_data": base64.b64encode(image_bytes).decode('utf-8')}
    except Exception as e:
        app.logger.error("Image transport store failed: %s", e)
        return jsonify({
            "error": "Failed to store uploaded image for processing",
            "image_id": image_id,
        }), 500

    def publish_all(channel):
        for service_name in service_names:
            queue_name = config.get_queue_name(f'primary.{service_name}')
            declare_queue(channel, queue_name)
            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps({
                    "image_id":       image_id,
                    "image_filename": image_filename,
                    "image_width":    normalized_width,
                    "image_height":   normalized_height,
                    "original_image_width": original_width,
                    "original_image_height": original_height,
                    "submitted_at":   datetime.now().isoformat(),
                    "trace_id":       trace_id,
                    "service_name":   service_name,
                    "queue_name":     queue_name,
                    "tier":           tier,
                    **image_transport,
                }),
                properties=pika.BasicProperties(delivery_mode=2),
            )

    try:
        publish_all(get_channel())
    except Exception as e:
        # Connection may have been silently dropped (TCP reset). Force a fresh
        # connection and retry once before giving up.
        app.logger.warning("Queue publish failed (%s), reconnecting and retrying...", e)
        try:
            publish_all(_new_rabbit_channel())
        except Exception as e2:
            app.logger.error("Queue publish failed after reconnect: %s", e2)
            return jsonify({
                "error":    "Failed to publish to processing queues",
                "image_id": image_id,
            }), 500

    return jsonify({
        "image_id":           image_id,
        "trace_id":           trace_id,
        "services_submitted": service_names,
        "image_width":        normalized_width,
        "image_height":       normalized_height,
        "original_image_width": original_width,
        "original_image_height": original_height,
        "normalized_image_width": normalized_width,
        "normalized_image_height": normalized_height,
    }), 202


@app.route('/status/<int:image_id>', methods=['GET'])
def status(image_id):
    """Check processing status and progressive results for an image."""
    try:
        db  = get_db()
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Image metadata
        cur.execute(
            """SELECT image_filename, image_group, services_submitted, image_created, tier,
                      original_image_width, original_image_height,
                      normalized_image_width, normalized_image_height
               FROM images WHERE image_id = %s""",
            (image_id,),
        )
        image_row = cur.fetchone()
        if not image_row:
            return jsonify({"error": "Image not found"}), 404

        services_submitted = image_row['services_submitted'] or []
        tier = image_row['tier'] or 'free'

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
                "status":          r['status'],
                "result_created":  r['result_created'].isoformat() if r['result_created'] else None,
                "processing_time": r['processing_time'],
            }

        services_pending = [s for s in services_submitted if s not in completed]
        total = len(services_submitted) if services_submitted else 0
        done  = len(completed)

        results_data = fetch_results(cur, image_id)

        vlm_short_names = {k.split('.', 1)[1] for k in config.get_services_by_type('vlm')}
        vlm_services = [s for s in services_submitted if s in vlm_short_names]

        primary_complete     = done == total and total > 0
        # Aggregate results (consensus, noun_consensus, etc.) live in
        # service_results after the fetch_results reshape — check both locations.
        _sr = results_data.get('service_results', {})
        _terminal = {'complete', 'failed', 'dead-lettered'}
        _failed_states = {'failed', 'dead-lettered'}
        _sd = results_data.get('service_dispatch', [])
        _current_sd = results_data.get('current_service_dispatch', [])
        _current_pp = results_data.get('current_postprocessing_events', [])
        _current_se = results_data.get('current_service_events', [])
        dispatches_pending = [
            row for row in _sd
            if row.get('status') == 'pending'
        ]
        dispatches_terminal = [
            row for row in _sd
            if row.get('status') in _terminal
        ]
        dispatches_failed = [
            row for row in _sd
            if row.get('status') in _failed_states
        ]

        result_failed = {
            service: data.get('data', {}).get('error_message')
            or data.get('data', {}).get('error')
            or data.get('data', {}).get('message')
            for service, data in _sr.items()
            if (data.get('status') or 'success') != 'success'
        }
        submitted_result_services = {
            service for service in _sr.keys()
            if service in services_submitted
        }
        services_pending = sorted({
            service for service in services_submitted
            if service not in submitted_result_services
        } | {
            row.get('service') for row in _current_sd
            if row.get('service') in services_submitted and row.get('status') == 'pending'
        })
        services_failed = {
            **{
                service: reason
                for service, reason in result_failed.items()
                if service in services_submitted
            },
            **{
                row.get('service'): row.get('failed_reason')
                for row in _current_sd
                if row.get('service') in services_submitted and row.get('status') in _failed_states
            },
        }
        downstream_pending = sorted({
            row.get('service') for row in _current_sd
            if row.get('service') not in services_submitted and row.get('status') == 'pending'
        } | {
            row.get('service') for row in _current_se
            if row.get('event_type') == 'enqueued'
        } | {
            row.get('service') for row in _current_pp
            if row.get('event_type') == 'enqueued'
        })
        downstream_failed = sorted({
            service for service in _sr.keys()
            if service not in services_submitted and (_sr.get(service, {}).get('status') or 'success') != 'success'
        } | {
            row.get('service') for row in _current_sd
            if row.get('service') not in services_submitted and row.get('status') in _failed_states
        } | {
            row.get('service') for row in _current_se
            if row.get('event_type') == 'failed'
        } | {
            row.get('service') for row in _current_pp
            if row.get('event_type') == 'failed'
        })

        service_events_pending = [
            row for row in _current_se
            if row.get('event_type') == 'enqueued'
        ]
        service_events_failed = [
            row for row in _current_se
            if row.get('event_type') == 'failed'
        ]
        postprocessing_pending = [
            row for row in _current_pp
            if row.get('event_type') == 'enqueued'
        ]
        postprocessing_failed = [
            row for row in _current_pp
            if row.get('event_type') == 'failed'
        ]

        is_complete = (
            len(_sd) > 0
            and len(dispatches_pending) == 0
            and len(service_events_pending) == 0
            and len(postprocessing_pending) == 0
        )

        return jsonify({
            "image_id":               image_id,
            "image_filename":         image_row['image_filename'],
            "image_group":            image_row['image_group'],
            "image_created":          image_row['image_created'].isoformat() if image_row['image_created'] else None,
            "original_image_width":   image_row['original_image_width'],
            "original_image_height":  image_row['original_image_height'],
            "normalized_image_width": image_row['normalized_image_width'],
            "normalized_image_height": image_row['normalized_image_height'],
            "services_submitted":     services_submitted,
            "vlm_services":           vlm_services,
            "services_completed":     completed,
            "services_pending":       services_pending,
            "services_failed":        services_failed,
            "progress":               f"{done}/{total}",
            "is_complete":            is_complete,
            "primary_complete":       primary_complete,
            "downstream_pending":     downstream_pending,
            "downstream_failed":      downstream_failed,
            "dispatches_total":       len(_sd),
            "dispatches_terminal":    len(dispatches_terminal),
            "dispatches_pending":     dispatches_pending,
            "dispatches_failed":      dispatches_failed,
            "postprocessing_events_total": len(results_data.get('postprocessing_events', [])),
            "postprocessing_events_pending": postprocessing_pending,
            "postprocessing_events_failed": postprocessing_failed,
            "service_events_total": len(results_data.get('service_events', [])),
            "service_events_pending": service_events_pending,
            "service_events_failed": service_events_failed,
            "content_analysis_complete": _sr.get('content_analysis') is not None,
            "noun_consensus_complete":    _sr.get('noun_consensus') is not None,
            "verb_consensus_complete":    _sr.get('verb_consensus') is not None,
            "sam3_complete":              results_data.get('sam3') is not None,
            "caption_summary_complete":   _sr.get('caption_summary') is not None,
            **results_data,
        })

    except Exception as e:
        app.logger.error("Status query failed: %s", e)
        return jsonify({"error": "Failed to query status"}), 500


@app.route('/results/<int:image_id>', methods=['GET'])
def results(image_id):
    """Get full analysis results for an image."""
    try:
        db  = get_db()
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Verify image exists
        cur.execute(
            """SELECT image_filename, image_group, services_submitted, image_created,
                      original_image_width, original_image_height,
                      normalized_image_width, normalized_image_height
               FROM images WHERE image_id = %s""",
            (image_id,),
        )
        image_row = cur.fetchone()
        if not image_row:
            return jsonify({"error": "Image not found"}), 404

        results_data = fetch_results(cur, image_id)

        return jsonify({
            "image_id":           image_id,
            "image_filename":     image_row['image_filename'],
            "image_group":        image_row['image_group'],
            "image_created":      image_row['image_created'].isoformat() if image_row['image_created'] else None,
            "original_image_width": image_row['original_image_width'],
            "original_image_height": image_row['original_image_height'],
            "normalized_image_width": image_row['normalized_image_width'],
            "normalized_image_height": image_row['normalized_image_height'],
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
    port  = int(os.getenv('API_PORT', 9999))
    debug = os.getenv('API_DEBUG', 'false').lower() == 'true'

    # Establish connections before accepting requests. Fail fast — there is no
    # point starting if the DB or queue are unreachable. Mid-operation drops are
    # still handled by the reconnect logic in get_db() / get_channel().
    try:
        get_db()
    except Exception as e:
        app.logger.error("Cannot connect to database at startup: %s", e)
        sys.exit(1)
    try:
        get_channel()
    except Exception as e:
        app.logger.error("Cannot connect to RabbitMQ at startup: %s", e)
        sys.exit(1)

    app.run(host='0.0.0.0', port=port, debug=debug)

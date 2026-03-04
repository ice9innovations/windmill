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

import imagehash
import ssl
import pika
import psycopg2
import psycopg2.extras
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from datetime import datetime, timezone
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Add workers directory to path for service_config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'workers'))
from service_config import get_service_config

register_heif_opener()  # adds HEIC/HEIF support to PIL globally
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

# Longest dimension (px) to normalize to before queuing; resize is in-memory only.
MAX_IMAGE_DIMENSION = int(os.getenv('MAX_IMAGE_DIMENSION', 2048))

SERVICE_CONFIG_PATH = os.getenv('SERVICE_CONFIG_PATH', 'service_config.yaml')
config = get_service_config(SERVICE_CONFIG_PATH)

_VALID_TIERS = frozenset({'free', 'basic', 'premium', 'cloud'})

# Age (seconds) after which a pending service_dispatch row with no result is considered stale.
# Override per-service via environment variables; defaults apply if not set.
_STALE_THRESHOLDS = {
    'sam3':            int(os.getenv('STALE_SAM3_SEC',            '600')),
    'face':            int(os.getenv('STALE_FACE_SEC',            '120')),
    'pose':            int(os.getenv('STALE_POSE_SEC',            '120')),
    'harmony':         int(os.getenv('STALE_HARMONY_SEC',         '60')),
    'consensus':       int(os.getenv('STALE_CONSENSUS_SEC',       '60')),
    'noun_consensus':  int(os.getenv('STALE_NOUN_CONSENSUS_SEC',  '60')),
    'verb_consensus':  int(os.getenv('STALE_VERB_CONSENSUS_SEC',  '60')),
    'caption_summary': int(os.getenv('STALE_CAPTION_SUMMARY_SEC', '120')),
}
_STALE_DEFAULT_SEC = int(os.getenv('STALE_DEFAULT_SEC', '300'))

# Result tables to query for read-time reconciliation of system worker dispatch rows.
# Keyed by service name; value is the SQL to check for any result for a given image_id.
_SYSTEM_RECONCILE_QUERIES = {
    'harmony':        "SELECT 1 FROM merged_boxes   WHERE image_id = %s LIMIT 1",
    'consensus':      "SELECT 1 FROM consensus      WHERE image_id = %s LIMIT 1",
    'noun_consensus': "SELECT 1 FROM noun_consensus WHERE image_id = %s LIMIT 1",
    'verb_consensus': "SELECT 1 FROM verb_consensus WHERE image_id = %s LIMIT 1",
    'caption_summary':"SELECT 1 FROM caption_summary WHERE image_id = %s LIMIT 1",
}

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
# Image helpers
# ---------------------------------------------------------------------------

def validate_and_normalize_image(image_bytes):
    """Validate image bytes and resize if necessary. No disk I/O ever.

    PIL's verify() performs an integrity check but closes the stream and
    invalidates the Image object, so we re-open from the original bytes
    for actual processing.

    Returns (normalized_bytes, width, height).
    Raises ValueError with a safe message if the bytes are not a valid image.
    """
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        raise ValueError("Invalid or corrupt image")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception:
        raise ValueError("Failed to decode image")

    original_format = img.format or 'JPEG'

    # Normalize EXIF orientation so all downstream workers (VLMs, SAM3, etc.)
    # receive pixels in display orientation. Without this, a portrait phone
    # photo stored as landscape pixels with an EXIF rotation tag would cause
    # bounding box coordinates to be in the raw (unrotated) space while the
    # browser displays the image rotated — producing misaligned overlays.
    transposed = ImageOps.exif_transpose(img)
    orientation_changed = transposed is not img
    img = transposed
    width, height = img.size

    # Formats that downstream workers (VLMs, SAM3, etc.) cannot handle.
    # Always re-encode these to JPEG regardless of image dimensions.
    _WEB_SAFE = {'JPEG', 'PNG', 'WEBP'}
    needs_transcode = original_format not in _WEB_SAFE

    if max(width, height) <= MAX_IMAGE_DIMENSION and not needs_transcode and not orientation_changed:
        return image_bytes, width, height

    # Resize if needed — all in memory, no temp files
    if max(width, height) > MAX_IMAGE_DIMENSION:
        ratio    = MAX_IMAGE_DIMENSION / max(width, height)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        img      = img.resize(new_size, Image.LANCZOS)
        width, height = img.size

    save_format = original_format if original_format in _WEB_SAFE else 'JPEG'
    if save_format == 'JPEG' and img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    buf = io.BytesIO()
    img.save(buf, format=save_format)
    app.logger.info(
        "%s -> %s %dx%d",
        original_format, save_format, width, height,
    )
    return buf.getvalue(), width, height


def compute_phash(image_bytes):
    """Compute a 64-bit perceptual hash of the image. Returns a 16-char hex string."""
    img = Image.open(io.BytesIO(image_bytes))
    return str(imagehash.phash(img))


def resolve_services(services_param, tier='free'):
    """Resolve a comma-separated service list, or return tier-appropriate primary services."""
    primary   = config.get_services_by_category('primary')
    available = sorted(name.split('.', 1)[1] for name in primary.keys())

    if not services_param:
        tier_services = config.get_services_by_tier(tier)
        return sorted(
            name.split('.', 1)[1]
            for name in tier_services.keys()
            if name.startswith('primary.')
        ), None

    requested = [s.strip() for s in services_param.split(',')]
    invalid   = [s for s in requested if s not in available]
    if invalid:
        return None, f"Unknown services: {', '.join(invalid)}. Available: {', '.join(available)}"
    return requested, None


def _compute_expected_downstream(services_submitted):
    """Determine which downstream services are expected based on submitted primary services.

    Returns a dict of {downstream_name: True/False} indicating whether each
    downstream service is expected to eventually produce a result for this image.
    Uses service_config.yaml definitions so the logic adapts when services change.
    """
    if not services_submitted:
        return {}

    has_consensus_service = any(
        config.should_trigger_consensus(f'primary.{s}')
        for s in services_submitted
    )

    vlm_services = [
        s for s in services_submitted
        if config.is_vlm_service(f'primary.{s}')
    ]
    has_vlm = len(vlm_services) > 0
    has_multi_vlm = len(vlm_services) >= 2

    return {
        'consensus':        has_consensus_service,
        'content_analysis': has_consensus_service,
        'noun_consensus':   has_vlm,
        'verb_consensus':   has_vlm,
        'sam3':             has_vlm,
        'caption_summary':  has_multi_vlm,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/analyze', methods=['POST'])
def analyze():
    """Submit a single image for analysis.

    Accepts multipart/form-data with a 'file' field containing an image.
    URL-based submission is not supported — send the image bytes directly.

    Optional form fields:
      - services:     comma-separated service list (default: tier-appropriate primary services)
      - image_group:  group tag for the image (default: 'api')
      - tier:         customer tier — free, basic, premium, cloud (default: 'free')
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
    services_param = request.form.get('services')
    image_group    = request.form.get('image_group', 'api')
    tier           = request.form.get('tier', 'free')

    if tier not in _VALID_TIERS:
        return jsonify({"error": f"Invalid tier '{tier}'. Valid tiers: {', '.join(sorted(_VALID_TIERS))}"}), 400

    if not image_bytes:
        return jsonify({"error": "Empty image data"}), 400

    # Validate and normalize — entirely in memory, never touches disk
    try:
        image_bytes, width, height = validate_and_normalize_image(image_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Perceptual hash — computed from the normalized image
    phash = None
    try:
        phash = compute_phash(image_bytes)
    except Exception as e:
        app.logger.warning("Perceptual hash computation failed: %s", e)

    # Validate services
    service_names, err = resolve_services(services_param, tier)
    if err:
        return jsonify({"error": err}), 400

    # Register image metadata — bytes are never stored anywhere
    try:
        db  = get_db()
        cur = db.cursor()
        cur.execute(
            """INSERT INTO images (image_filename, image_group, services_submitted, image_phash, tier)
               VALUES (%s, %s, %s, %s, %s)
               RETURNING image_id""",
            (image_filename, image_group, service_names, phash, tier),
        )
        image_id = cur.fetchone()[0]
        # Record pending dispatch for all primary services — best-effort tracking.
        # Failures here must never abort image submission.
        try:
            cur.execute(
                "INSERT INTO service_dispatch (image_id, service) SELECT %s, unnest(%s::text[])",
                (image_id, service_names),
            )
        except Exception as e:
            app.logger.warning("Failed to record service_dispatch entries: %s", e)
    except Exception as e:
        app.logger.error("Database insert failed: %s", e)
        return jsonify({"error": "Failed to register image in database"}), 500

    # Publish to queues
    trace_id = str(uuid.uuid4())
    b64_data = base64.b64encode(image_bytes).decode('utf-8')

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
                    "image_data":     b64_data,
                    "image_width":    width,
                    "image_height":   height,
                    "submitted_at":   datetime.now().isoformat(),
                    "trace_id":       trace_id,
                    "service_name":   service_name,
                    "queue_name":     queue_name,
                    "tier":           tier,
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
        "image_width":        width,
        "image_height":       height,
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
            "data":            r['data'],
            "processing_time": r['processing_time'],
            "result_created":  r['result_created'].isoformat() if r['result_created'] else None,
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
            "consensus_data":    consensus_row['consensus_data'],
            "processing_time":   consensus_row['processing_time'],
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
    content_row      = cur.fetchone()
    content_analysis = dict(content_row) if content_row else None
    if content_analysis and content_analysis.get('created'):
        content_analysis['created'] = content_analysis['created'].isoformat()

    # Postprocessing — also resolve source_bbox for canvas coordinate transforms.
    # For merged_box rows: bbox lives in merged_boxes.merged_data.merged_bbox.
    # For SAM3 rows (merged_box_id IS NULL): cluster_id in data.cluster_id lets us
    # look up the instance bbox from sam3_results.
    cur.execute(
        """SELECT p.service, p.merged_box_id, p.data, p.processing_time,
                  mb.merged_data->'merged_bbox' AS source_bbox
           FROM postprocessing p
           LEFT JOIN merged_boxes mb ON mb.merged_id = p.merged_box_id
           WHERE p.image_id = %s AND p.status = 'success'""",
        (image_id,),
    )
    postprocessing = [dict(r) for r in cur.fetchall()]

    # Noun consensus
    cur.execute(
        """SELECT nouns, category_tally, services_present, service_count, created_at, updated_at
           FROM noun_consensus WHERE image_id = %s""",
        (image_id,),
    )
    noun_consensus_row = cur.fetchone()
    noun_consensus     = None
    if noun_consensus_row:
        all_nouns      = noun_consensus_row['nouns'] or []
        consensus_nouns = [n for n in all_nouns if n.get('confidence', 0) > 0.5 or n.get('promoted', False)]
        noun_consensus = {
            "nouns":            consensus_nouns,
            "nouns_all":        all_nouns,
            "category_tally":   noun_consensus_row['category_tally'] or [],
            "services_present": noun_consensus_row['services_present'],
            "service_count":    noun_consensus_row['service_count'],
            "created_at":       noun_consensus_row['created_at'].isoformat() if noun_consensus_row['created_at'] else None,
            "updated_at":       noun_consensus_row['updated_at'].isoformat() if noun_consensus_row['updated_at'] else None,
        }

    # SAM3 segmentation results
    cur.execute(
        """SELECT nouns_queried, data, instance_count, processing_time, created_at, updated_at
           FROM sam3_results WHERE image_id = %s""",
        (image_id,),
    )
    sam3_row    = cur.fetchone()
    sam3_results = None
    if sam3_row:
        sam3_results = {
            "nouns_queried":  sam3_row['nouns_queried'],
            "results":        sam3_row['data'],
            "instance_count": sam3_row['instance_count'],
            "processing_time": sam3_row['processing_time'],
            "created_at":     sam3_row['created_at'].isoformat() if sam3_row['created_at'] else None,
            "updated_at":     sam3_row['updated_at'].isoformat() if sam3_row['updated_at'] else None,
        }

    # Verb consensus
    cur.execute(
        """SELECT verbs, svo_triples, services_present, service_count, created_at, updated_at
           FROM verb_consensus WHERE image_id = %s""",
        (image_id,),
    )
    verb_consensus_row = cur.fetchone()
    verb_consensus     = None
    if verb_consensus_row:
        verb_consensus = {
            "verbs":            verb_consensus_row['verbs'] or [],
            "svo_triples":      verb_consensus_row['svo_triples'] or {},
            "services_present": verb_consensus_row['services_present'],
            "service_count":    verb_consensus_row['service_count'],
            "created_at":       verb_consensus_row['created_at'].isoformat() if verb_consensus_row['created_at'] else None,
            "updated_at":       verb_consensus_row['updated_at'].isoformat() if verb_consensus_row['updated_at'] else None,
        }

    # Caption summary
    cur.execute(
        """SELECT summary_caption, model, services_present, service_count, created_at, updated_at
           FROM caption_summary WHERE image_id = %s""",
        (image_id,),
    )
    caption_summary_row = cur.fetchone()
    caption_summary     = None
    if caption_summary_row:
        caption_summary = {
            "summary_caption":  caption_summary_row['summary_caption'],
            "model":            caption_summary_row['model'],
            "services_present": caption_summary_row['services_present'],
            "service_count":    caption_summary_row['service_count'],
            "created_at":       caption_summary_row['created_at'].isoformat() if caption_summary_row['created_at'] else None,
            "updated_at":       caption_summary_row['updated_at'].isoformat() if caption_summary_row['updated_at'] else None,
        }

    # Service dispatch — one row per (service, cluster_id), latest by dispatched_at.
    # DISTINCT ON gives the most recent dispatch per (service, cluster_id) pair so that
    # re-harmonization re-dispatches show the current status, not stale earlier ones.
    cur.execute(
        """SELECT DISTINCT ON (service, cluster_id)
                  service, cluster_id, status, dispatched_at
           FROM service_dispatch
           WHERE image_id = %s
           ORDER BY service, cluster_id NULLS LAST, dispatched_at DESC""",
        (image_id,),
    )
    service_dispatch = []
    for r in cur.fetchall():
        row = dict(r)
        if row.get('dispatched_at'):
            row['dispatched_at'] = row['dispatched_at'].isoformat()
        service_dispatch.append(row)

    # Read-time reconciliation: result tables are authoritative.
    # A dispatch row stuck in 'pending' despite a completed result means the worker
    # crashed between writing the result and updating the dispatch row — treat as complete.
    pending = [r for r in service_dispatch if r['status'] == 'pending']
    if pending:
        # Primary services (image-level, not sam3): check results table
        primary_pending_svcs = [
            r['service'] for r in pending
            if r['cluster_id'] is None and r['service'] != 'sam3'
        ]
        if primary_pending_svcs:
            cur.execute(
                "SELECT DISTINCT service FROM results WHERE image_id = %s AND service = ANY(%s)",
                (image_id, primary_pending_svcs),
            )
            results_found = {row['service'] for row in cur.fetchall()}
            for r in pending:
                if r['service'] in results_found and r['cluster_id'] is None:
                    r['status'] = 'complete'

        # SAM3: check sam3_results table
        sam3_pending = [r for r in pending if r['service'] == 'sam3' and r['status'] == 'pending']
        if sam3_pending:
            cur.execute("SELECT 1 FROM sam3_results WHERE image_id = %s LIMIT 1", (image_id,))
            if cur.fetchone():
                for r in sam3_pending:
                    r['status'] = 'complete'

        # Face/pose (bbox-level): check postprocessing table by (service, cluster_id)
        bbox_pending = [r for r in pending if r['cluster_id'] is not None and r['status'] == 'pending']
        if bbox_pending:
            pending_cluster_ids = [r['cluster_id'] for r in bbox_pending]
            pending_services    = list({r['service'] for r in bbox_pending})
            cur.execute(
                """SELECT DISTINCT service, data->>'cluster_id' AS cluster_id
                   FROM postprocessing
                   WHERE image_id = %s AND service = ANY(%s)
                     AND data->>'cluster_id' = ANY(%s)""",
                (image_id, pending_services, pending_cluster_ids),
            )
            post_found = {(row['service'], row['cluster_id']) for row in cur.fetchall()}
            for r in bbox_pending:
                if (r['service'], r['cluster_id']) in post_found:
                    r['status'] = 'complete'

        # System workers (harmony, consensus, noun/verb consensus, caption_summary):
        # check each worker's result table to catch the crash-between-result-and-update case.
        system_pending = [
            r for r in pending
            if r['cluster_id'] is None
            and r['service'] in _SYSTEM_RECONCILE_QUERIES
            and r['status'] == 'pending'
        ]
        for r in system_pending:
            cur.execute(_SYSTEM_RECONCILE_QUERIES[r['service']], (image_id,))
            if cur.fetchone():
                r['status'] = 'complete'

    # Stale detection: pending + no result + past age threshold → 'stale' at read time.
    # This is never written to the DB; it is a computed property for the caller.
    now_utc = datetime.now(timezone.utc)
    for r in service_dispatch:
        if r['status'] != 'pending':
            continue
        threshold = _STALE_THRESHOLDS.get(r['service'], _STALE_DEFAULT_SEC)
        dispatched_str = r.get('dispatched_at')
        if dispatched_str:
            try:
                dispatched_at = datetime.fromisoformat(dispatched_str)
                if dispatched_at.tzinfo is None:
                    dispatched_at = dispatched_at.replace(tzinfo=timezone.utc)
                if (now_utc - dispatched_at).total_seconds() > threshold:
                    r['status'] = 'stale'
            except Exception:
                pass

    # Resolve source_bbox for SAM3-dispatched postprocessing rows (merged_box_id IS NULL).
    # These rows store cluster_id = "sam3:noun:idx" in data; use it to look up the
    # instance bbox from sam3_results so the console can draw pose skeletons.
    if sam3_results:
        sam3_data = sam3_results.get('results') or {}
        for row in postprocessing:
            if row.get('source_bbox') is not None:
                continue  # already resolved via merged_boxes JOIN
            cluster_id = (row.get('data') or {}).get('cluster_id', '')
            if not cluster_id.startswith('sam3:'):
                continue
            parts = cluster_id.split(':')
            if len(parts) == 3:
                _, noun, idx_str = parts
                try:
                    inst = sam3_data.get(noun, {}).get('instances', [])[int(idx_str)]
                    row['source_bbox'] = inst.get('bbox')
                except (IndexError, ValueError, TypeError):
                    pass

    return {
        "service_results":  service_results,
        "merged_boxes":     merged_boxes,
        "consensus":        consensus,
        "content_analysis": content_analysis,
        "postprocessing":   postprocessing,
        "noun_consensus":   noun_consensus,
        "sam3":             sam3_results,
        "verb_consensus":   verb_consensus,
        "caption_summary":  caption_summary,
        "service_dispatch": service_dispatch,
    }


@app.route('/status/<int:image_id>', methods=['GET'])
def status(image_id):
    """Check processing status and progressive results for an image."""
    try:
        db  = get_db()
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
                "status":          r['status'],
                "result_created":  r['result_created'].isoformat() if r['result_created'] else None,
                "processing_time": r['processing_time'],
            }

        services_pending = [s for s in services_submitted if s not in completed]
        total = len(services_submitted) if services_submitted else 0
        done  = len(completed)

        results_data = _fetch_results(cur, image_id)

        vlm_short_names = {k.split('.', 1)[1] for k in config.get_services_by_type('vlm')}
        vlm_services = [s for s in services_submitted if s in vlm_short_names]

        primary_complete     = done == total and total > 0
        expected_downstream  = _compute_expected_downstream(services_submitted)
        downstream_pending   = [
            svc for svc, expected in expected_downstream.items()
            if expected and results_data.get(svc) is None
        ]
        is_complete = primary_complete and len(downstream_pending) == 0

        return jsonify({
            "image_id":               image_id,
            "image_filename":         image_row['image_filename'],
            "image_group":            image_row['image_group'],
            "image_created":          image_row['image_created'].isoformat() if image_row['image_created'] else None,
            "services_submitted":     services_submitted,
            "vlm_services":           vlm_services,
            "services_completed":     completed,
            "services_pending":       services_pending,
            "progress":               f"{done}/{total}",
            "is_complete":            is_complete,
            "primary_complete":       primary_complete,
            "downstream_pending":     downstream_pending,
            "consensus_complete":     results_data['consensus'] is not None,
            "content_analysis_complete": results_data['content_analysis'] is not None,
            "noun_consensus_complete":    results_data['noun_consensus'] is not None,
            "verb_consensus_complete":    results_data['verb_consensus'] is not None,
            "sam3_complete":              results_data['sam3'] is not None,
            "caption_summary_complete":   results_data['caption_summary'] is not None,
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
            """SELECT image_filename, image_group, services_submitted, image_created
               FROM images WHERE image_id = %s""",
            (image_id,),
        )
        image_row = cur.fetchone()
        if not image_row:
            return jsonify({"error": "Image not found"}), 404

        results_data = _fetch_results(cur, image_id)

        return jsonify({
            "image_id":           image_id,
            "image_filename":     image_row['image_filename'],
            "image_group":        image_row['image_group'],
            "image_created":      image_row['image_created'].isoformat() if image_row['image_created'] else None,
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

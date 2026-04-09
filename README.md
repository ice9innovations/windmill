# Windmill

> "Windmill or no windmill, he said, life would go on as it had always gone on — that is, badly."

Distributed ML image processing pipeline for Animal Farm. Transforms monolithic API calls into a continuous queue-based pipeline capable of processing image streams at scale across 10+ ML services.

Related docs:

- [INSTALLATION.md](/home/sd/windmill/INSTALLATION.md)
- [DEPLOYMENT.md](/home/sd/windmill/DEPLOYMENT.md)
- [CONTRIBUTING.md](/home/sd/windmill/CONTRIBUTING.md)
- [RUNBOOK.md](/home/sd/windmill/RUNBOOK.md)
- [docs/workflow.md](/home/sd/windmill/docs/workflow.md)

---

## How It Works

Images are submitted to the API (or directly via the producer) and dispatched into RabbitMQ queues. Workers consume from their queues, call the corresponding ML service, and write results to PostgreSQL. Downstream workers trigger progressively as prerequisite results arrive, so noun grounding, segmentation, verb extraction, caption synthesis, and safety analysis can start before the full graph is complete.

```
Submit image
    │
    ├─→ [blip] [moondream] [ollama] [haiku] [gemini] [gpt_nano] [qwen]   ← VLM / semantic
    ├─→ [colors] [ocr] [nudenet] [metadata]                                ← specialized
    └─→ [yolo_v8]                                                          ← spatial (bbox)
            │
            ├─→ harmony_worker → merged_boxes
            │       └─→ [colors_post] [face] [pose]   ← postprocessing on each bbox
            │
            └─→ noun_consensus_worker → noun_consensus
                    ├─→ verb_consensus
                    ├─→ florence2_grounding_worker → florence2_grounding
                    ├─→ sam3_worker → sam3_results
                    └─→ caption_summary_worker → caption_summary
                                                    └─→ content_analysis_worker → content_analysis
```

---

## Services

### Primary Services (run on full image)

| Service | Port | Type | Tiers |
|---------|------|------|-------|
| blip | 7777 | semantic, vlm | basic, premium |
| florence2 | 7803 | semantic, vlm | basic, premium, batch |
| colors | 7770 | colors | free, basic, premium, batch |
| metadata | 7781 | metadata | free, basic, premium, batch |
| nsfw2 | 7774 | nsfw | free, basic, premium, batch |
| ocr | 7775 | specialized | basic, premium, batch |
| qr | 7801 | specialized | basic, premium, batch |
| nudenet | 7789 | specialized | free, basic, premium, batch |
| moondream | 7795 | semantic, vlm | basic, premium |
| ollama | 7782 | semantic, vlm | basic, premium |
| haiku | 7797 | semantic, vlm | premium, batch |
| gemini | 7767 | semantic, vlm | premium, batch |
| gpt_nano | 7800 | semantic, vlm | premium, batch |
| xai | 7805 | semantic, vlm | premium, batch |
| qwen | 7796 | semantic, vlm | basic, premium |
| yolo_v8 | 7773 | spatial | free, basic, premium, batch |

### Postprocessing Services (run on cropped bbox regions)

| Service | Port | Type | Trigger |
|---------|------|------|---------|
| colors_post | 7770 | colors | all bboxes |
| face | 7772 | specialized | person bboxes only |
| pose | 7786 | specialized | person bboxes only |
| caption_score | 7778 | caption_score | after each VLM result |

### System Services (internal pipeline workers)

| Service | Role |
|---------|------|
| harmony | IoU clustering across spatial results → merged_boxes |
| florence2_grounding | progressive noun grounding requests driven by noun_consensus |
| noun_consensus | Noun extraction + synonym collapse → noun_consensus |
| verb_consensus | Verb/SVO extraction → verb_consensus |
| sam3 | SAM3 segmentation for detected nouns → sam3_results |
| caption_summary | LLM synthesis of all VLM captions → caption_summary |
| content_analysis | safety/scene synthesis over noun/verb consensus, captions, and nudenet |
| rembg | Background removal → used by downstream consumers |

---

## Tier System

Services are gated by tier. The `tier` field is set at image submission time.

| Tier | Included services |
|------|------------------|
| free | colors, metadata, nsfw2, nudenet, yolo_v8 |
| basic | free + blip, florence2, moondream, ollama, ocr, qr, qwen |
| premium | basic + haiku, gemini, gpt_nano, xai |
| batch | colors, metadata, nsfw2, ocr, qr, nudenet, florence2, haiku, gemini, gpt_nano, xai, yolo_v8 |

Tiers are defined in `service_config.yaml`. Adding a new tier there automatically makes it valid — no code change required.

---

## Configuration

### `service_config.yaml`

Defines all services: ports, endpoints, service types, queue names, and tier assignments. Consumed by workers, api.py, and the producer.

```yaml
services:
  primary:
    blip:
      queue_name: blip
      host: localhost
      port: 7777
      endpoint: /v3/analyze
      service_type: semantic, vlm
      tier: [basic, premium]
  postprocessing:
    ...
  system:
    ...
```

### `.env`

Copy `.env.example` to `.env` and fill in your values.

```bash
# Database
DB_HOST=your-db-host
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=...
DB_SSLMODE=prefer

# RabbitMQ
QUEUE_HOST=your-queue-host
QUEUE_PORT=5671
QUEUE_SSL=true
QUEUE_USER=your_queue_user
QUEUE_PASSWORD=...

# Optional: Valkey-backed image transport
IMAGE_STORE_MODE=inline   # inline | valkey
VALKEY_HOST=
VALKEY_PORT=6379
VALKEY_SSL=true
VALKEY_USERNAME=
VALKEY_PASSWORD=
VALKEY_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
VALKEY_IMAGE_TTL_SECONDS=90
VALKEY_CROP_TTL_SECONDS=90
VALKEY_SOCKET_CONNECT_TIMEOUT_SECONDS=3
VALKEY_SOCKET_TIMEOUT_SECONDS=3
VALKEY_HEALTH_CHECK_INTERVAL_SECONDS=30
VALKEY_KEEPALIVE_PING_SECONDS=15

# API port (default 9999)
API_PORT=9997
```

All VLM services (haiku, gemini, gpt_nano) communicate through Animal Farm service endpoints — no API keys are configured in windmill.

`IMAGE_STORE_MODE=inline` is the default. Set `IMAGE_STORE_MODE=valkey` to
store uploaded images and bbox crops in Valkey with a short TTL and pass only
opaque refs through RabbitMQ. That mode requires a TLS-enabled Valkey server
with ACL auth. See [docs/valkey-image-store.md](/home/sd/windmill/docs/valkey-image-store.md).

---

## API

Start the API server:
```bash
python api.py
# or set API_PORT in .env
```

### POST /analyze

Submit an image for processing.

```bash
curl -X POST \
  -F "file=@photo.jpg" \
  -F "tier=basic" \
  http://localhost:9997/analyze
```

Form fields:
- `file` — image file (required)
- `tier` — free / basic / premium / batch (default: free)
- `services` — comma-separated override list (default: all services for the tier)
- `image_group` — group tag (default: api)

Response `202`:
```json
{
  "image_id": 517,
  "trace_id": "a1b2c3d4-...",
  "services_submitted": ["blip", "colors", "moondream", "yolo_v8"],
  "image_width": 1280,
  "image_height": 720
}
```

### GET /status/{image_id}

Poll for progressive results. Returns all completed data immediately, with completeness flags for each downstream step.

```bash
curl http://localhost:9997/status/517
```

```json
{
  "image_id": 517,
  "progress": "3/4",
  "is_complete": false,
  "primary_complete": false,
  "downstream_pending": ["noun_consensus", "caption_summary"],
  "services_submitted": ["blip", "colors", "moondream", "yolo_v8"],
  "services_completed": {"blip": {...}, "colors": {...}},
  "services_pending": ["moondream", "yolo_v8"],
  "noun_consensus_complete": false,
  "verb_consensus_complete": false,
  "sam3_complete": false,
  "caption_summary_complete": false,
  "content_analysis_complete": false,
  "service_results": {...},
  "merged_boxes": [...],
  "postprocessing": [...],
  "service_dispatch": [
    {"service": "blip", "status": "complete", "dispatched_at": "..."},
    {"service": "moondream", "status": "pending", "dispatched_at": "..."}
  ]
}
```

### GET /results/{image_id}

Full results without status metadata. Use when processing is confirmed complete.

### GET /workflow

Returns the machine-readable Windmill workflow contract: symbolic predicates, downstream stages, and trigger sources.

This endpoint is intended for API consumers and other repos that need to reason about dependency relationships without reimplementing `core.workflow.compute_expected_downstream()`.

---

## Batch Processing

```bash
# Submit all unprocessed images for all primary services
./producer.sh --limit 1000

# Submit a specific image group
./producer.sh --group coco2017 --limit 10000

# Resume from a specific image_id
./producer.sh --start-id 50000 --limit 5000
```

The producer reads from the `images` table and dispatches to all primary service queues for the configured tier.

---

## Worker Management

```bash
# Start all workers
./windmill.sh start

# Start a specific worker
./windmill.sh start harmony_worker
./windmill.sh start noun_consensus_worker

# Stop all workers
./windmill.sh stop

# Stop a specific worker
./windmill.sh stop blip_worker

# Restart all workers
./windmill.sh reset

# Check status (reads from worker_registry table)
./windmill.sh status
```

Worker state is persisted in `.windmill_state`. Workers auto-register in the `worker_registry` table and send heartbeats every 30 seconds. The `registry_sweeper_worker` marks stale workers (no heartbeat for 90s) as offline.

---

## Job Lifecycle

Every primary service dispatch is tracked in `service_dispatch`:

| Status | Meaning |
|--------|---------|
| pending | Dispatched to queue, not yet processed |
| complete | Worker wrote a result successfully |
| failed | Worker reported a terminal failure |

---

## Observability

### Check service dispatch status
```sql
SELECT service, status, count(*)
FROM service_dispatch
GROUP BY service, status
ORDER BY service, status;
```

### Check queue depths
```bash
# Via RabbitMQ management UI
http://your-host:15672

# Via CLI
rabbitmqctl list_queues name messages
```

### Check processing rates
```sql
-- Results per service
SELECT service, COUNT(*), COUNT(*) FILTER (WHERE status = 'success') AS success_count
FROM results GROUP BY service ORDER BY service;

-- Harmonized boxes
SELECT COUNT(*) FROM merged_boxes;

-- Postprocessing results
SELECT service, COUNT(*) FROM postprocessing GROUP BY service;
```

### Worker logs
```bash
tail -f logs/harmony_worker.log
tail -f logs/noun_consensus_worker.log
```

---

## Queue Failures

Queues are declared directly without a paired DLQ worker. Terminal worker failures are written explicitly to `service_dispatch.failed_reason` and the message is acknowledged by the worker.

Use the RabbitMQ management UI to inspect live queues directly if needed.

---

## Database Schema

Full schema in `db/db.sql`. Key tables:

| Table | Pattern | Contents |
|-------|---------|---------|
| images | INSERT | Source image metadata, tier, services_submitted |
| results | INSERT | Per-service ML results |
| service_dispatch | UPDATE | Job lifecycle per service (pending → complete/failed) |
| merged_boxes | DELETE+INSERT | Harmonized bounding boxes |
| postprocessing | INSERT | Per-bbox service results |
| noun_consensus | UPSERT | Extracted nouns, categories, confidence |
| verb_consensus | UPSERT | Extracted verbs, SVO triples |
| sam3_results | UPSERT | SAM3 segmentation per noun |
| caption_summary | UPSERT | LLM-synthesized caption |
| content_analysis | UPSERT | Scene classification |
| worker_registry | UPSERT | Live worker heartbeats |

> **Note**: `db/db.sql` may not match the live schema exactly. Always check live column names before writing queries.

---

## Multi-Node Deployment

Workers are stateless — any machine with `.env` credentials and a `*_worker.py` matching the worker name can join the fleet. RabbitMQ load-balances across all consumers of the same queue automatically.

```bash
# Machine A: heavy semantic workers
./windmill.sh start blip_worker moondream_worker ollama_worker

# Machine B: spatial + postprocessing
./windmill.sh start yolo_v8_worker harmony_worker face_worker pose_worker

# Machine C: system workers
./windmill.sh start noun_consensus_worker caption_summary_worker
```

---

## Code Layout

```
api.py                    Flask gateway (submit, status, results)
core/
  image.py                Image validation and phash (no Flask dependency)
  dispatch.py             Service resolution and downstream computation
  results.py              All DB result queries
workers/
  base_worker.py          BaseWorker — shared queue/DB/HTTP plumbing
  service_config.py       ServiceConfig class, get_service_config()
  harmony_worker.py       IoU clustering and postprocessing dispatch
  noun_consensus_worker.py Noun extraction, verb/SVO extraction, category tally
  noun_extractor.py       spaCy-based noun/verb extraction
  noun_utils.py           ConceptNet synonym collapse
  caption_summary_worker.py LLM caption synthesis
  sam3_worker.py          SAM3 segmentation dispatcher
  registry_sweeper_worker.py Stale worker cleanup
  producer.py             Batch job submission
service_config.yaml       Service definitions, tiers, ports
db/db.sql                 Reference schema (not safe to run on live DB)
utils/                    CLI utilities (list_dlqs, requeue_from_dlq, etc.)
```

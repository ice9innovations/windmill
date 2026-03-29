# Windmill Installation

This guide is for bringing up Windmill itself.

If you need the broader multi-repo deployment picture, start with [DEPLOYMENT.md](/home/sd/windmill/DEPLOYMENT.md). Windmill is an orchestrator; it is not useful by itself unless compatible ML services are reachable from `service_config.yaml`.

## Prerequisites

Required:

- Python 3.10+
- PostgreSQL 13+
- RabbitMQ 3.x
- `pgvector` extension in PostgreSQL
- one or more Animal Farm-compatible ML service endpoints

Recommended:

- RabbitMQ management plugin
- systemd for API and worker supervision
- Ansible for multi-node operations

Optional but relevant:

- ConceptNet load for noun-consensus synonym collapse
- GPU-backed model hosts for heavier services

## What Windmill Depends On

Windmill expects:

- PostgreSQL for `images`, `results`, aggregate tables, lifecycle state, and worker registry
- RabbitMQ for primary and downstream queueing
- HTTP ML services for actual inference

Minimal end-to-end setup:

- one VLM-style primary service
- one spatial primary service
- PostgreSQL
- RabbitMQ
- a worker node running the corresponding workers plus downstream system workers

## 1. Clone And Create A Virtualenv

```bash
git clone <your-windmill-remote>
cd windmill

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure `.env`

Copy `.env.example` to `.env` and fill in values for your environment.

Typical keys:

```bash
QUEUE_HOST=your-rabbitmq-host
QUEUE_PORT=5671
QUEUE_SSL=true
QUEUE_USER=your-user
QUEUE_PASSWORD=your-password

DB_HOST=your-postgres-host
DB_NAME=your-db-name
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_SSLMODE=prefer

WORKER_PREFETCH_COUNT=1
REQUEST_TIMEOUT=30
MAX_RETRIES=3
RETRY_DELAY=5
QUEUE_MESSAGE_TTL_MS=120000
LOG_LEVEL=INFO
```

Important notes:

- the API and workers share the same queue and DB connectivity settings
- `service_config.yaml` is separate from `.env`; it controls service endpoints and tier membership
- producers must publish `trace_id`; Windmill uses it for primary-result idempotency
- `IMAGE_STORE_MODE=inline` is the default and requires no extra infrastructure

### Optional: Valkey-backed image transport

Set this only if you want uploads and bbox crops stored out-of-band in Valkey
instead of being embedded in RabbitMQ messages.

```bash
IMAGE_STORE_MODE=valkey
VALKEY_HOST=images.ice9.ai
VALKEY_PORT=6379
VALKEY_SSL=true
VALKEY_USERNAME=windmill
VALKEY_PASSWORD=...
VALKEY_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
VALKEY_IMAGE_TTL_SECONDS=90
VALKEY_CROP_TTL_SECONDS=90
```

Recommended Valkey server properties for this mode:

- TLS enabled
- ACL auth enabled
- persistence disabled: `save ""` and `appendonly no`
- `maxmemory-policy noeviction`

Quick connectivity check from a Windmill node:

```bash
export REDISCLI_AUTH='your-secret'
valkey-cli --tls -h "$VALKEY_HOST" -p "$VALKEY_PORT" \
  --cacert "$VALKEY_CA_CERTS" \
  --user "$VALKEY_USERNAME" \
  ping
unset REDISCLI_AUTH
```

See [docs/valkey-image-store.md](/home/sd/windmill/docs/valkey-image-store.md)
for the dedicated mode guide.

## 3. Prepare PostgreSQL

Create the database and role referenced in `.env`, then apply the schema:

```bash
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f db/db.sql
```

Schema notes:

- `db/db.sql` creates `pgvector` if available
- `results.source_trace_id` is part of the primary retry-safety design
- `service_dispatch` is the authoritative per-service lifecycle table
- `worker_registry` stores live worker heartbeats

## 4. Load ConceptNet If You Need Noun Consensus

If your deployment uses noun consensus, load ConceptNet edges:

```bash
./utils/load_conceptnet.sh /path/to/conceptnet-assertions-5.7.0.csv.gz
```

This populates `conceptnet_edges`, which noun consensus uses to build its in-memory synonym graph.

## 5. Configure `service_config.yaml`

Edit [service_config.yaml](/home/sd/windmill/service_config.yaml) so every enabled service points at a real host/port/endpoint.

This file controls:

- service catalog
- queue names
- service types
- tier membership
- which services are consensus-eligible

Do not use it to describe internal downstream dependency logic. That is documented separately in [docs/workflow.md](/home/sd/windmill/docs/workflow.md).

## 6. Verify Core Connectivity

Before starting workers, verify:

- PostgreSQL is reachable from the Windmill host
- RabbitMQ is reachable from the Windmill host
- Valkey is reachable from the Windmill host when `IMAGE_STORE_MODE=valkey`
- each configured service endpoint is reachable from the worker host

At minimum:

- `psql` can connect
- the API can connect to RabbitMQ
- the API and workers can `PING` Valkey when `IMAGE_STORE_MODE=valkey`
- workers can hit their configured ML service hosts

## 7. Start Windmill

### Local worker control

```bash
./windmill.sh start
./windmill.sh status
```

### Local API

```bash
python api.py
```

If you run the API under systemd, verify with:

```bash
systemctl status windmill-api
```

## 8. Submit A Test Image

```bash
curl -X POST \
  -F "file=@photo.jpg" \
  -F "tier=basic" \
  http://127.0.0.1:9997/analyze
```

Then verify:

- an `images` row exists
- primary `results` rows are being written
- `results.source_trace_id` is populated on those primary rows
- `service_dispatch` moves to terminal states
- downstream stages expected for the tier complete

## 9. Verify Operations

Use these checks:

- `./windmill.sh status`
- RabbitMQ management UI
- `logs/`
- `SELECT * FROM worker_registry WHERE status = 'online'`
- `SELECT * FROM service_dispatch WHERE image_id = <id>`

## 10. Multi-Node Notes

For distributed workers:

- keep `.env` aligned across nodes that share the same DB and broker
- assign service hosts in `service_config.yaml` so each worker can reach its service
- use [ansible/inventory.example.yml](/home/sd/windmill/ansible/inventory.example.yml) and [ansible/README.md](/home/sd/windmill/ansible/README.md) for node management

Windmill is designed so most workers can have multiple copies running at once.

## 11. Common Installation Pitfalls

- Windmill starts, but nothing processes
  - workers are not actually online
  - RabbitMQ queues have no consumers
  - service hosts in `service_config.yaml` are unreachable
- API accepts images, but results never appear
  - workers are offline
  - RabbitMQ publish succeeded but no consumers are running
- noun consensus behaves oddly
  - ConceptNet was never loaded
- tier behavior looks wrong
  - `service_config.yaml` tier membership differs from assumptions in old notes or scripts

## Docker Quickstart

For local development only:

```bash
docker compose up --build
docker compose exec windmill bash -lc 'psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f db/db.sql'
```

Then submit a test image and verify the same lifecycle checks as above.

## Dataset Seeding By URL

Windmill can process images referenced by URL instead of local file paths.

CSV import helper:

```bash
DB_HOST=... DB_NAME=... DB_USER=... DB_PASSWORD=... \
python utils/import_images_csv.py --csv /path/to/images.csv
```

CSV format:

```text
image_id,image_filename,image_url,image_group
1,000000000001.jpg,https://host/coco/train2017/000000000001.jpg,coco2017
```

This is useful for external datasets and remote worker fleets where local file mounting is undesirable.

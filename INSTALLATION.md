## Windmill Installation (Deployment‑Agnostic)

This guide sets up Windmill without assumptions about specific hosts or Raspberry Pi hardware.

### 1) Prerequisites
- Python 3.10+
- RabbitMQ (with management plugin enabled)
- PostgreSQL 13+

Optional:
- GPU services depending on which ML backends you run

### 2) Environment Variables (.env)
Create `.env` in repo root:
```
QUEUE_HOST=your-rabbitmq-host
QUEUE_USER=your-user
QUEUE_PASSWORD=your-password

DB_HOST=your-postgres-host
DB_NAME=your-db-name
DB_USER=your-db-user
DB_PASSWORD=your-db-password

# Optional tuning
WORKER_PREFETCH_COUNT=1
REQUEST_TIMEOUT=30
MAX_RETRIES=3
RETRY_DELAY=5
QUEUE_MESSAGE_TTL_MS=120000
LOG_LEVEL=INFO
```

### 3) Install Python Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4) Prepare PostgreSQL
- Create database and user matching `.env`
- Apply schema:
```bash
psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f db/db.sql
```

### 5) RabbitMQ
Ensure the management plugin is enabled and credentials match `.env`.
DLQ/TTL are auto-declared by workers and producer.

### 6) Configure Services
Edit `service_config.yaml` to point to your service hosts/ports. The file is categorized by `primary`, `postprocessing`, and `system` components.

### 7) Start Workers
```bash
./windmill.sh start             # start all detected workers
./windmill.sh status            # verify they are running
```

### 8) Submit Jobs
```bash
./producer.sh --limit 1000      # submits to the safe default primary set
./producer.sh --group demo --limit 100
```

### 9) Monitor
- RabbitMQ UI (Queues, DLQs)
- Logs in `logs/`
- Database queries in README

### 10) Common Variations
- Single‑machine: run RabbitMQ, PostgreSQL, ML services, and workers on one host
- Split infra: run RabbitMQ and PostgreSQL on managed/cloud services, workers on compute nodes
- GPU nodes: run heavy ML services on separate machines; workers connect via host/port in `service_config.yaml`

### 11) Upgrades and Safe Reprocessing

## Docker Quickstart (Local Dev)

Spin up RabbitMQ + Postgres + Windmill:
```bash
docker compose up --build
```

Then apply the DB schema inside the windmill container:
```bash
docker compose exec windmill bash -lc 'psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -f db/db.sql'
```

Submit jobs:
```bash
docker compose exec windmill ./producer.sh --limit 100
```

View RabbitMQ UI at http://localhost:15672 (user/pass from QUEUE_USER/QUEUE_PASSWORD env or defaults).

### Using External Datasets (URLs)

Windmill prefers `image_url` over local file paths. You can seed images that live on any reachable HTTP(S) host and avoid mounting files:

- Add images via CSV importer (below) or direct DB insert with `image_url` populated
- Producer will fetch by URL, extract dimensions, and base64-encode into jobs

CSV import helper:
```bash
DB_HOST=... DB_NAME=... DB_USER=... DB_PASSWORD=... \
python utils/import_images_csv.py --csv /path/to/images.csv
```

CSV format (header row required):
```
image_id,image_filename,image_url,image_group
1,000000000001.jpg,https://host/coco/train2017/000000000001.jpg,coco2017
```

- Use DLQs to isolate repeated failures
- Use the provided utils to list/requeue from DLQs
- For harmonization logic changes, re-run harmony then consensus as documented in README



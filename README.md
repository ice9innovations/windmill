# Windmill - Distributed ML Processing Pipeline for Animal Farm

"Windmill or no windmill, he said, life would go on as it had always gone on — that is, badly."

A complete distributed processing system. This system transforms Animal Farm from monolithic API to continuous processing pipeline capable of infinite-scale image streams across the Animal Farm machine learning services.

## Overview

This system delivers the complete vision of distributed ML processing with three key innovations:

1. **Complete Service Coverage**: 15 ML services (BLIP, CLIP, Colors, Detectron2, Face, Xception, Metadata, OCR, NSFW2, Ollama, Pose, RT-DETR, YOLOv8, YOLO_365, YOLO_OI7) plus Caption Scoring postprocessing
2. **Progressive Harmonization**: Immediate results after ANY service completes, with automatic re-harmonization as more services finish
3. **Distributed Spatial Processing**: Specialized postprocessing workers for bbox-level face detection, pose estimation, and color analysis

The system processes images through the full ML pipeline while solving the specialized challenge of **bounding box harmonization** from detection services.

### Key Innovation: Progressive Harmonization

Instead of waiting for all bbox services to complete, the system **harmonizes immediately after ANY service completes**, then re-harmonizes as more services finish:

1. **RT-DETR completes** → Harmonize with RT-DETR results alone
2. **YOLOv8 completes** → Re-harmonize with RT-DETR + YOLOv8 results  
3. **Detectron2 completes** → Re-harmonize with all three services

This approach is **fault-tolerant** (works if services fail) and **performance-optimized** (immediate results that improve over time).

## Architecture Components

### 1. Message Queue Infrastructure
- **RabbitMQ** 
- **PostgreSQL** 
- **Queue-per-service** pattern (`queue_colors`, `queue_yolov8`, etc.)

### 2. Core Workers

#### Service-Specific ML Workers
- **Dedicated workers** for each service (blip_worker.py, clip_worker.py, colors_worker.py, etc.)
- **BaseWorker inheritance** (`base_worker.py`) provides shared database, queue, and HTTP functionality
- **Service-specific classes** extend BaseWorker with minimal service-specific logic

#### Bounding Box Harmonizer (`harmony_worker.py`) 
- **Purpose**: Harmonize bbox results from yolov8, rtdetr, detectron2
- **Trigger**: Queue-based processing via `queue_harmony`
- **Algorithm**: Cross-service IoU clustering with democratic filtering  
- **Output**: Harmonized bounding boxes stored in `merged_boxes` table
- **Dispatch**: Crops bounding boxes and dispatches to postprocessing queues

#### Consensus Worker (`consensus_worker.py`)
- **Purpose**: Calculate voting consensus using V3 voting algorithm
- **Trigger**: Queue-based processing via `queue_consensus`
- **Algorithm**: Democratic voting with evidence weighting from spatial and semantic services
- **Output**: Consensus results stored in `consensus` table

#### Postprocessing Workers (Queue-Based Spatial Analysis)
- **Bbox Colors Worker** (`bbox_colors_worker.py`): Processes `queue_bbox_colors` for cropped bounding box color analysis
- **Bbox Face Worker** (`bbox_face_worker.py`): Processes `queue_bbox_face` for person bounding boxes only  
- **Bbox Pose Worker** (`bbox_pose_worker.py`): Processes `queue_bbox_pose` for person bounding boxes only
- **Architecture**: Pure queue-based workers consuming from dedicated postprocessing queues
- **Processing**: POST cropped images to existing ML services (colors:7770, face:7772, pose:7786)
- **Output**: Results stored in `postprocessing` table with service-specific naming

### 3. Database Schema

**Complete schema documentation:** See `database_schema.sql` for detailed table definitions, indexes, and foreign key relationships.

**Key Tables:**
- `results` - Core ML service results (INSERT-only pattern)
- `merged_boxes` - Harmonized bounding boxes (DELETE+INSERT pattern) 
- `consensus` - Voting consensus results (DELETE+INSERT pattern)
- `postprocessing` - Spatial analysis results (INSERT-only pattern)
- `images` - Source images with URLs/paths for recovery operations

**Key Design Decisions:**
- **DELETE+INSERT** for merged_boxes/consensus (clean JOINs, atomic re-harmonization)
- **INSERT-only** for postprocessing (no update conflicts, parallel worker safety)
- **Foreign key constraints** ensure referential integrity across the pipeline
- **Monitoring views** provide real-time processing status
- **Performance indexes** optimize worker query patterns

## Configuration

### Service Configuration (`service_config.json`)
Maps service names to ports, endpoints, and categorization:
```json
{
  "services": {
    "blip": {
      "port": 7777,
      "endpoint": "/v3/analyze", 
      "category": "primary",
      "description": "Image captioning with emoji mapping",
      "enable_consensus_triggers": true
    },
    "face": {
      "port": 7772,
      "endpoint": "/v3/analyze",
      "category": "spatial_only", 
      "description": "Face detection",
      "enable_consensus_triggers": true
    }
  }
}
```

**Service Categories:**
- **`primary`**: Whole-image ML services, included in default job submission
- **`spatial_only`**: Bbox-region services (face/pose), handled by postprocessing workers
- **`postprocessing`**: Caption scoring and other post-ML analysis services

### Worker Configuration
Workers load configuration from `service_config.json` for service definitions and `.env` files for credentials:

**Service Configuration (`service_config.json`):**
```json
{
  "services": {
    "yolov8": {
      "host": "localhost", 
      "port": 7773,
      "endpoint": "/v3/analyze",
      "category": "primary",
      "service_type": "spatial",
      "enable_triggers": true
    }
  }
}
```

**Infrastructure Configuration (`.env`):**
```bash
# Infrastructure endpoints
QUEUE_HOST=192.168.0.122
DB_HOST=192.168.0.121

# Credentials
QUEUE_USER=animal_farm
QUEUE_PASSWORD=your_secure_queue_password
DB_USER=animal_farm_user
DB_PASSWORD=your_secure_db_password
```

## Usage

### 1. Single Image API

A Flask API for submitting individual images and retrieving progressive results.

```bash
# Start the API server (default port 9999, configurable via API_PORT)
python api.py
```

#### Submit an image

File upload:
```bash
curl -X POST -F "image=@photo.jpg" http://localhost:9999/analyze
```

URL-based:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image_url": "http://example.com/photo.jpg"}' \
  http://localhost:9999/analyze
```

Response (202):
```json
{
  "image_id": 391978,
  "trace_id": "a1b2c3d4-...",
  "services_submitted": ["blip", "colors", "detectron2", "..."]
}
```

Optional parameters:
- `services` - comma-separated list (default: all primary services)
- `image_group` - group tag (default: `"api"`)

#### Poll for status and progressive results

```bash
curl http://localhost:9999/status/391978
```

Returns status metadata and full results as they arrive. Clients can start using partial results immediately (e.g. show captions while waiting for object detection):

```json
{
  "image_id": 391978,
  "progress": "10/14",
  "is_complete": false,
  "services_pending": ["clip", "nsfw2", "ollama", "rtmdet"],
  "services_completed": {"blip": {"status": "success", "..."}, "..."},
  "harmony_complete": true,
  "consensus_complete": true,
  "content_analysis_complete": true,
  "service_results": {"blip": {"data": {"..."}, "..."}, "..."},
  "merged_boxes": ["..."],
  "consensus": {"..."},
  "content_analysis": {"..."},
  "postprocessing": ["..."]
}
```

#### Fetch final results

```bash
curl http://localhost:9999/results/391978
```

Returns the full analysis output without status metadata. Use this when processing is complete and you just need the data.

### 2. Batch Processing
```bash
# Submit jobs to all PRIMARY services (safe default)
./producer.sh --limit 100000

# Process a specific image group
./producer.sh --group coco2017 --limit 10000
```

Note: Selecting individual services has been intentionally removed to prevent data inconsistencies. The producer now targets the primary, safe set by design.

### 2. Worker Management
```bash
# Start all workers
./windmill.sh start

# Start specific workers  
./windmill.sh start bbox_merger
./windmill.sh start blip
./windmill.sh start consensus

# Stop all workers
./windmill.sh stop

# Stop specific worker
./windmill.sh stop ollama

# Reset all workers (stop and start)
./windmill.sh reset

# Check worker status
./windmill.sh status
```

**Available Workers (Dynamically Detected):**
- **ML Service Workers**: blip, clip, colors, detectron2, xception, metadata, nsfw2, ocr, ollama, rtdetr, yolov8, yolo_365, yolo_oi7
- **Processing Workers**: harmony (harmonization), consensus (voting), caption_score
- **Postprocessing Workers**: bbox_colors, bbox_face, bbox_pose (spatial analysis)

**Note**: The `windmill.sh` script automatically detects all available worker files and can start/stop/reset individual workers or all workers. ML service workers will only start successfully if the corresponding services are running on their configured ports.

### 3. Monitor Progress
```sql
-- Check processing status
SELECT service, COUNT(*), 
       COUNT(*) FILTER (WHERE status = 'success') as success_count
FROM results GROUP BY service;

-- Check harmonization status
SELECT COUNT(*) as total_merged_boxes FROM merged_boxes;

-- Check spatial enrichment status
SELECT COUNT(*) as total_enrichments FROM postprocessing;
```

## Key Features

### Fault Tolerance
- Services can fail independently without blocking others
- Workers automatically retry failed jobs
- Bbox harmonization works with any subset of services (1, 2, or 3)

### Horizontal Scaling  
- Add workers by copying `.env` files to new machines
- RabbitMQ automatically load-balances across workers
- No coordination between workers needed

### Progressive Processing
- **Immediate results** that improve over time
- **No blocking** - each service processes independently  
- **Automatic re-harmonization** when new services complete

### Spatial Intelligence
- **True bbox-level analysis** via in-memory cropping
- **Face detection** on person bounding boxes → 🙂 emoji results
- **Color analysis** on all bounding boxes → Full Prismacolor palettes
- **Zero file I/O** during processing (legal/performance benefits)

## Complete Processing Pipeline Flow

```
1. Images → Primary Service Queues → Distributed ML Workers
    ↓        ↓         ↓            ↓            ↓
  BLIP   CLIP   Colors  Detectron2  Xception  Metadata  
  OCR    NSFW2  Ollama  RT-DETR     YOLOv8    YOLO_365  YOLO_OI7
                     ↓
2. All Primary Results → PostgreSQL results table
                     ↓  
3. Bbox Services → queue_harmony → harmony_worker → merged_boxes table
                     ↓ (Crops bboxes and dispatches to postprocessing queues)
4. ALL Services → queue_consensus → consensus_worker → consensus table (V3 Voting Algorithm)
                     ↓
5. Cropped Bboxes → Distributed Postprocessing Queues → Postprocessing Workers:
   • queue_bbox_colors → bbox_colors_worker → Colors Service (port 7770) → postprocessing table  
   • queue_bbox_face → bbox_face_worker → Face Service (port 7772) → postprocessing table
   • queue_bbox_pose → bbox_pose_worker → Pose Service (port 7786) → postprocessing table
```

**Complete Pipeline Example:**
- T+0s: Colors, CLIP complete → queue_consensus → Consensus worker runs V3 voting on 2 services
- T+3s: YOLOv8 completes → queue_bbox_merge → Bbox merger harmonizes YOLOv8 alone, dispatches to postprocessing queues
- T+5s: RT-DETR, BLIP complete → queue_bbox_merge → Bbox merger re-harmonizes YOLOv8+RT-DETR
- T+8s: All primary services complete → Final bbox harmonization and consensus across all services
- T+10s: Postprocessing queues → Workers process cropped regions in parallel → Face/pose/color analysis

## Performance Characteristics

- **Throughput**: Scales linearly with worker count
- **Latency**: Results available immediately after first service  
- **Resource usage**: GPU services can run on dedicated hardware
- **Storage**: ~1KB per result, ~10KB per merged box, ~5KB per enrichment

## Deployment

See INSTALLATION.md for a setup guide (RabbitMQ/PostgreSQL anywhere, single‑machine or split infra).

See RUNBOOK.md for operations (restart order, DLQ triage, reprocessing).

## Deployment Patterns

### Single Machine Development
```bash
# Terminal 1: Start ML services (blip:7777, clip:7788, yolov8:7773, face:7772, etc.)
# Terminal 2: ./windmill.sh start harmony consensus
# Terminal 3: ./windmill.sh start blip clip colors yolov8
# Terminal 4: ./windmill.sh start bbox_colors bbox_face bbox_pose
```

### Cloud Scaling
- **Queue workers** scale elastically based on queue depth
- **GPU workers** can use spot instances for cost optimization
- **Storage tier** can use managed PostgreSQL for reliability

## Troubleshooting

### Common Issues

**Worker stops processing:**
```bash
# Check queue has jobs
curl -u animal_farm:your_secure_queue_password [server]:15672/api/queues

# Check service is responding  
curl http://localhost:7770/analyze
```

**Bbox merger not running:**
```sql
-- Check for new bbox results
SELECT COUNT(*) FROM results WHERE service IN ('yolov8','rtdetr','detectron2');

-- Check merger processed them
SELECT COUNT(*) FROM merged_boxes;
```

**Spatial enrichment failing:**
```bash  
# Check face/colors services
curl -X POST -F "file=@test.jpg" http://localhost:7772/analyze
curl -X POST -F "file=@test.jpg" http://localhost:7770/analyze
```

### Monitoring Commands
```bash
# Queue depths
rabbitmqctl list_queues name messages

# Processing rates  
watch 'psql -c "SELECT COUNT(*) FROM results"'

# Worker logs
tail -f /var/log/animal-farm/workers.log
```

### Dead Letter Queues (DLQ)

All queues are created with a paired DLQ and sane defaults (TTL/max length). Each queue `X` has a dead-letter queue `X.dlq`.

Inspect DLQs via RabbitMQ UI (Queues tab) or CLI:
```bash
# List DLQs quickly via utility
QUEUE_HOST=... QUEUE_USER=... QUEUE_PASSWORD=... \
python utils/list_dlqs.py
```

Requeue items from a DLQ back to the original queue:
- Recommended: Use the RabbitMQ UI (Queues → <queue>.dlq → Get messages → Requeue to original)
- Or use the utility script:

```bash
QUEUE_HOST=... QUEUE_USER=... QUEUE_PASSWORD=... \
python utils/requeue_from_dlq.py queue_harmony --limit 500
```

Notes:
- DLQs accumulate messages that exceeded retries/TTL or were negatively acknowledged repeatedly.
- Investigate root-cause via worker logs before mass requeueing.

#### Using RabbitMQ Web UI

1) Inspect DLQs
- Go to Queues → filter for `.dlq` queues → click a DLQ
- Check “Messages” panel for total, ready/unacked, and rates

2) Peek messages
- In the DLQ page, click “Get messages”, set a small count (e.g., 10)
- Ack mode:
  - “Ack & remove” to drain samples
  - “Requeue” puts them back into the same DLQ (not original queue)

3) Purge a DLQ
- DLQ page → “Purge Messages” (deletes all messages in the DLQ)

4) Bulk requeue DLQ → original queue (UI shovel)
- Enable shovels (one-time on the RabbitMQ host):
  ```bash
  rabbitmq-plugins enable rabbitmq_shovel rabbitmq_shovel_management
  ```
- UI → Admin → Shovels → Add new:
  - Source: Queue = <queue>.dlq
  - Destination: Queue = <queue>
  - Prefetch = 100 (example), “Delete when” = idle
  - Create; it will drain DLQ to the original queue and self-delete when idle

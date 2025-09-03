# Windmill - Distributed ML Processing Pipeline for Animal Farm

"Windmill or no windmill, he said, life would go on as it had always gone on — that is, badly."

A complete distributed processing system. This system transforms Animal Farm from monolithic API to continuous processing pipeline capable of infinite-scale image streams across the Animal Farm machine learning services.

## Overview

This system delivers the complete vision of distributed ML processing with three key innovations:

1. **Complete Service Coverage**: All services (BLIP, CLIP, Colors, Detectron2, Face, Xception, Metadata, OCR, NSFW2, Ollama, Pose, RT-DETR, YOLOv8, Caption Scoring) 
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
- **RabbitMQ** on k2.local (192.168.0.122)
- **PostgreSQL** on k1.local (192.168.0.121)
- **Queue-per-service** pattern (`queue_colors`, `queue_yolov8`, etc.)

### 2. Core Workers

#### Service-Specific ML Workers
- **Dedicated workers** for each service (blip_worker.py, clip_worker.py, colors_worker.py, etc.)
- **Base worker class** (`base_worker.py`) provides shared functionality
- **Service-specific logic** handles unique requirements per ML service

#### Bounding Box Merger (`bbox_merger_worker.py`) 
- **Purpose**: Harmonize bbox results from yolov8, rtdetr, detectron2
- **Trigger**: Runs after ANY bbox service completes (timestamp-based detection)
- **Algorithm**: Cross-service IoU clustering with democratic filtering
- **Output**: Harmonized bounding boxes stored in `merged_boxes` table

#### Consensus Worker (`consensus_worker.py`)
- **Purpose**: Calculate voting consensus using V3 voting algorithm
- **Trigger**: Runs after any successful service completion
- **Algorithm**: Democratic voting with evidence weighting
- **Output**: Consensus results stored in `consensus` table

#### Postprocessing Workers (Distributed Spatial Analysis)
- **Bbox Colors Worker** (`bbox_colors_worker.py`): Color analysis on ALL cropped bounding boxes
- **Bbox Face Worker** (`bbox_face_worker.py`): Face detection on person bounding boxes only  
- **Bbox Pose Worker** (`bbox_pose_worker.py`): Pose estimation on person bounding boxes only
- **Method**: bbox_merger_worker crops regions and dispatches to specialized queues
- **Innovation**: Parallel spatial processing instead of sequential bottleneck
- **Output**: Results stored in `postprocessing` table with service-specific naming

### 3. Database Schema

```sql
-- Core ML service results
results(image_id, service, data, status, processing_time, result_created, worker_id)

-- Harmonized bounding boxes (DELETE+INSERT pattern)
merged_boxes(image_id, source_result_ids, merged_data, status, created, worker_id) 

-- Voting consensus (DELETE+INSERT pattern)  
consensus(image_id, consensus_data, processing_time, consensus_created, worker_id)

-- Spatial analysis results (INSERT-only pattern)
postprocessing(image_id, merged_box_id, service, data, status, result_created, worker_id)

-- Source images
images(image_id, image_filename, image_path, image_url, image_created)
```

**Database Setup:**
```bash
# Create database and tables
psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f schema.sql
```

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

### Worker Configuration (`.env` files)
Each worker loads configuration from `.env`:
```bash
# Which service to run
SERVICE_NAME=yolov8

# Infrastructure  
QUEUE_HOST=192.168.0.122
DB_HOST=192.168.0.121

# Credentials
QUEUE_USER=animal_farm
QUEUE_PASSWORD=your_secure_queue_password
DB_USER=animal_farm_user
DB_PASSWORD=your_secure_db_password

# Worker settings
WORKER_ID=worker_yolov8_box1
LOG_LEVEL=INFO
```

## Usage

### 1. Submit Jobs - Complete ML Pipeline
```bash
# Process 100K images through all PRIMARY services (excludes face/pose - they run via postprocessing)
python producer.py --services all --limit 100000

# Process through all services including spatial_only (face/pose)
python producer.py --services full_catalog --limit 100000

# Submit jobs to specific services for testing
python producer.py --services blip,clip,colors,yolov8 --limit 1000

# Process specific image group
python producer.py --services all --group coco2017 --limit 10000

# List all available services and categories
python producer.py --list-services
```

**Service Categories:**
- **`all`/`primary`**: Main ML services that process whole images (excludes face/pose)
- **`spatial_only`**: Face/pose services (handled by postprocessing workers, not for direct submission)
- **`full_catalog`**: All services including spatial_only (use with caution)
- **Custom list**: Comma-separated service names for targeted testing

### 2. Worker Management
```bash
# Start all workers
./workers.sh start

# Start specific workers  
./workers.sh start bbox_merger
./workers.sh start blip
./workers.sh start consensus

# Stop all workers
./workers.sh stop

# Stop specific worker
./workers.sh stop ollama

# Restart all workers (useful after code changes)
./workers.sh restart

# Restart specific worker
./workers.sh restart blip

# Check worker status
./workers.sh status
```

**Available Workers (Dynamically Detected):**
- **ML Service Workers**: blip, clip, colors, detectron2, xception, metadata, nsfw2, ocr, ollama, rtdetr, yolov8
- **Processing Workers**: bbox_merger (harmonization), consensus (voting), caption_score
- **Postprocessing Workers**: bbox_colors, bbox_face, bbox_pose (spatial analysis)

**Note**: The `workers.sh` script automatically detects all available worker files and can start/stop/restart individual workers or all workers. ML service workers will only start successfully if the corresponding services are running on their configured ports.

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
  OCR    NSFW2  Ollama  RT-DETR     YOLOv8    Caption_Score
                     ↓
2. All Primary Results → PostgreSQL results table
                     ↓  
3. Bbox Services → bbox_merger_worker → merged_boxes table
                     ↓ (Crops bboxes and dispatches to postprocessing queues)
4. ALL Services → consensus_worker → consensus table (V3 Voting Algorithm)
                     ↓
5. Cropped Bboxes → Distributed Postprocessing Workers:
   • All boxes → bbox_colors_worker → Colors Service (port 7770) → postprocessing table  
   • Person boxes → bbox_face_worker → Face Service (port 7772) → postprocessing table
   • Person boxes → bbox_pose_worker → Pose Service (port 7786) → postprocessing table
```

**Complete Pipeline Example:**
- T+0s: Colors, CLIP complete → Consensus worker runs V3 voting on 2 services
- T+3s: YOLOv8 completes → Bbox merger harmonizes YOLOv8 alone, crops boxes, dispatches to postprocessing queues
- T+5s: RT-DETR, BLIP complete → Bbox merger re-harmonizes YOLOv8+RT-DETR, dispatches new boxes
- T+8s: All primary services complete → Final bbox harmonization, final consensus across all services
- T+10s: Postprocessing workers process cropped regions in parallel → Face/pose/color analysis

## Performance Characteristics

- **Throughput**: Scales linearly with worker count
- **Latency**: Results available immediately after first service  
- **Resource usage**: GPU services can run on dedicated hardware
- **Storage**: ~1KB per result, ~10KB per merged box, ~5KB per enrichment

## Deployment Patterns

### Single Machine Development
```bash
# Terminal 1: Start ML services (blip:7777, clip:7788, yolov8:7773, face:7772, etc.)
# Terminal 2: ./workers.sh start bbox_merger consensus
# Terminal 3: ./workers.sh start blip clip colors yolov8
# Terminal 4: ./workers.sh start bbox_colors bbox_face bbox_pose
```

### Pi Cluster Production
```bash
# k1.local (Pi with storage): PostgreSQL + consensus_worker  
# k2.local (Pi with SSD): RabbitMQ + bbox_merger_worker
# k3.local (Pi with GPU): yolov8/rtdetr services + workers
# k4.local (Pi with NPU): face/pose services + postprocessing workers
# Main box: Primary ML services + workers
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
curl -u animal_farm:your_secure_queue_password http://192.168.0.122:15672/api/queues

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

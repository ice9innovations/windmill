# Windmill - Distributed ML Processing Pipeline

"Windmill or no windmill, he said, life would go on as it had always gone on -- that is, badly"

A complete distributed processing system implementing the **EPIC: Distributed Queue-Based ML Processing Architecture**. This system transforms Animal Farm from monolithic API to continuous processing pipeline capable of infinite-scale image streams across 13 ML services.

## Overview

This system delivers the complete **EPIC vision** of distributed ML processing with three key innovations:

1. **Complete Service Coverage**: All 13 EPIC services (BLIP, CLIP, Colors, Detectron2, Face, Inception_v3, Metadata, OCR, NSFW2, Ollama, Pose, RT-DETR, YOLOv8) 
2. **Progressive Harmonization**: Immediate results after ANY service completes, with automatic re-harmonization as more services finish
3. **True Spatial Enrichment**: In-memory bbox cropping for genuine spatial analysis of detected objects

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

#### Generic ML Worker (`generic_worker.py`)
- Configurable via `.env` files for any of 13 services
- Processes images through ML services and stores results
- Supports both URL and file-based image processing

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

#### Spatial Enrichment Worker (`spatial_enrichment_worker.py`)
- **Purpose**: Add face/pose data for person boxes, colors for all boxes
- **Method**: Crops bbox regions in memory, POSTs to face/pose/colors services
- **Innovation**: True spatial analysis (not whole-image analysis)
- **Output**: Enrichment data stored in `postprocessing` table

### 3. Database Schema

```sql
-- Core ML service results
results(image_id, service, data, status, processing_time)

-- Harmonized bounding boxes (DELETE+INSERT pattern)
merged_boxes(image_id, source_result_ids, merged_data, status) 

-- Voting consensus (DELETE+INSERT pattern)  
consensus(image_id, consensus_data, processing_time)

-- Spatial enrichment results (INSERT-only pattern)
postprocessing(image_id, merged_box_id, service, data)
```

**Key Design Decisions:**
- **DELETE+INSERT** for merged_boxes/consensus (clean JOINs, no history)
- **INSERT-only** for postprocessing (no update conflicts, parallel workers)
- **Timestamp-based triggers** (no explicit coordination needed)

## Configuration

### Service Configuration (`service_config.json`)
Maps service names to ports and endpoints:
```json
{
  "services": {
    "colors": {"port": 7770, "endpoint": "/analyze"},
    "yolov8": {"port": 7773, "endpoint": "/analyze"}
  }
}
```

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
# Process 100K images through ALL 13 ML services (1.3M total jobs)
python generic_producer.py --services all --limit 100000

# Or specific services for testing
python generic_producer.py --services blip,clip,colors,yolov8 --limit 1000

# List all available services
python generic_producer.py --list-services
```

### 2. Run Workers - Distributed Across Infrastructure
```bash
# Complete ML service workers (13 services available)
SERVICE_NAME=blip python generic_worker.py       # Image captioning
SERVICE_NAME=clip python generic_worker.py       # Image classification
SERVICE_NAME=colors python generic_worker.py     # Color analysis
SERVICE_NAME=detectron2 python generic_worker.py # Object detection
SERVICE_NAME=face python generic_worker.py       # Face detection
SERVICE_NAME=inception_v3 python generic_worker.py # ImageNet classification
SERVICE_NAME=metadata python generic_worker.py   # EXIF extraction
SERVICE_NAME=ocr python generic_worker.py        # Text extraction
SERVICE_NAME=nsfw2 python generic_worker.py      # Content moderation
SERVICE_NAME=ollama python generic_worker.py     # LLM vision analysis
SERVICE_NAME=pose python generic_worker.py       # Pose estimation
SERVICE_NAME=rtdetr python generic_worker.py     # Transformer object detection
SERVICE_NAME=yolov8 python generic_worker.py     # Real-time object detection

# Post-processing workers (specialized)
python bbox_merger_worker.py         # Harmonizes object detection results
python consensus_worker.py           # V3 voting algorithm across ALL services
python spatial_enrichment_worker.py  # Spatial analysis on detected objects
```

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
1. Images → 13 Service Queues → Distributed ML Workers
    ↓        ↓         ↓            ↓
  BLIP   CLIP   Colors  Detectron2  Face  Inception_v3  Metadata  
  OCR    NSFW2  Ollama  RT-DETR     YOLOv8     Pose
                     ↓
2. All ML Results → PostgreSQL results table
                     ↓  
3. Bbox Services → bbox_merger_worker → merged_boxes table
                     ↓
4. ALL Services → consensus_worker → consensus table (V3 Voting Algorithm)
                     ↓
5. Merged Boxes → spatial_enrichment_worker → postprocessing table
```

**Complete Pipeline Example:**
- T+0s: Colors, CLIP complete → Consensus worker runs V3 voting on 2 services
- T+3s: YOLOv8 completes → Bbox merger harmonizes YOLOv8 alone, consensus re-runs on 3 services
- T+5s: RT-DETR, BLIP complete → Bbox merger re-harmonizes YOLOv8+RT-DETR, consensus on 5 services
- T+8s: All 13 services complete → Final bbox harmonization, final consensus across all services
- T+10s: Spatial enrichment crops person boxes → Face analysis + color analysis

## Performance Characteristics

- **Throughput**: Scales linearly with worker count
- **Latency**: Results available immediately after first service  
- **Resource usage**: GPU services can run on dedicated hardware
- **Storage**: ~1KB per result, ~10KB per merged box, ~5KB per enrichment

## Deployment Patterns

### Single Machine Development
```bash
# Terminal 1: Start services (colors, yolov8, face, etc.)
# Terminal 2: python generic_worker.py  
# Terminal 3: python bbox_merger_worker.py
# Terminal 4: python spatial_enrichment_worker.py
```

### Pi Cluster Production
```bash
# k1.local (Pi with storage): PostgreSQL + consensus_worker
# k2.local (Pi with SSD): RabbitMQ + spatial_enrichment_worker  
# k3.local (Pi with GPU): yolov8 service + worker
# k4.local (Pi with NPU): rtdetr service + worker
# Main box: All other services + workers + bbox_merger_worker
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

## Contributing

### Adding New Services
1. Add service definition to `service_config.json`
2. Configure service in single `.env` file
3. Add queue creation to `generic_producer.py`
4. Start worker: `SERVICE_NAME=newservice python generic_worker.py`

### Adding New Post-Processing Workers
1. Follow the timestamp-based trigger pattern
2. Use DELETE+INSERT for single-record tables
3. Use INSERT-only for multi-record tables  
4. Include proper error handling and logging

## Architecture Philosophy

This system embodies the **"Jimmy vs James" principle** from CONTRIBUTING.md:

- **Jimmy**: Builds complex coordination systems with locks, semaphores, and explicit messaging
- **James**: Uses simple timestamp comparison to let workers detect their own work

We chose the **James approach**: workers automatically detect when their results are stale and recompute. No coordination protocols, no deadlocks, no complexity - just elegant data dependencies.

The result is a system that **scales beautifully** because it has **no coordination overhead**.

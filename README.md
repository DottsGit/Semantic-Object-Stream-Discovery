# SOSD - Semantic Object Stream Discovery

An unsupervised vision pipeline that ingests video streams, discovers object types through clustering, and tracks flow statistics in real-time.

## Features

- **Multi-source video ingestion**: RTSP, HLS, YouTube, webcam, and local files
- **Unsupervised object clustering**: Uses DINOv2 features + HDBSCAN to automatically discover object categories
- **Real-time tracking**: SORT-based multi-object tracking with cluster assignment
- **Flow analytics**: Per-cluster flow rates, velocities, and statistics
- **GCP integration**: Pub/Sub events, BigQuery analytics, Cloud Run deployment
- **Live dashboard**: Streamlit-based visualization (demo mode)

## How It Works

```
Video Stream → Object Detection → DINOv2 Features → Clustering → Tracking → Analytics
                    (YOLO)          (unsupervised)    (HDBSCAN)    (SORT)
```

1. **Warmup Phase** (configurable, default 60s): Collects object samples and extracts DINOv2 embeddings
2. **Clustering Phase**: Runs unsupervised clustering (HDBSCAN) to discover natural object groupings
3. **Running Phase**: Real-time detection, classification by cluster, tracking, and flow analytics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourrepo/semantic-object-stream-discovery.git
cd semantic-object-stream-discovery

# Install dependencies
pip install -e .
```

### Run with a video file

```bash
python -m src.main path/to/video.mp4 --warmup 60
```

### Run with webcam

```bash
python -m src.main 0 --warmup 30
```

### Run with RTSP stream

```bash
python -m src.main "rtsp://camera-ip:554/stream" --warmup 120
```

### Run with YouTube stream

```bash
python -m src.main "https://youtube.com/watch?v=..." --warmup 60
```

### Vehicles only (highway traffic)

```bash
python -m src.main video.mp4 --vehicles-only --warmup 90
```

## CLI Options

```
python -m src.main [SOURCE] [OPTIONS]

Arguments:
  SOURCE          Video source (URL, file path, or webcam index)

Options:
  --type          Source type: auto, rtsp, hls, youtube, file, webcam
  --warmup        Warmup duration in seconds (default: 60)
  --clusters      Number of clusters (None = auto-detect)
  --min-cluster-size Min samples per cluster (default: 100)
  --device        Processing device: cuda, cpu, mps (default: cuda)
  --no-display    Disable visualization window
  --detector      YOLO model: yolov8n.pt, yolov8s.pt, etc.
  --vehicles-only Only detect vehicles (cars, trucks, buses, motorcycles)
```

## Configuration

Configuration is primarily handled via CLI arguments.
Default settings are defined in `src/main.py`.

The `config/settings.py` file contains schema definitions for future configuration expansion using environment variables, but currently `main.py` relies on command-line arguments.

## Dashboard

Run the Streamlit dashboard (Demo Mode):

```bash
streamlit run src/output/dashboard.py
```

Then open http://localhost:8501

## Docker

### Build and run locally

```bash
docker-compose up --build
```

### With GPU support

```bash
docker-compose -f docker-compose.yml up --build sosd-pipeline
```

## GCP Deployment

### Cloud Run (serverless, CPU)

```bash
./deploy/gcp-cloud-run.sh
```

### Compute Engine (GPU)

```bash
./deploy/gcp-compute-engine.sh
```

## Architecture

```
src/
├── ingestion/       # Video source handling (RTSP, HLS, YouTube, files)
├── detection/       # YOLOv8 object detection
├── features/        # DINOv2 feature extraction
├── clustering/      # HDBSCAN/K-means unsupervised clustering
├── tracking/        # SORT-based multi-object tracking
├── analytics/       # Flow statistics and metrics
├── output/          # Dashboard, Pub/Sub, BigQuery
└── main.py          # Main pipeline orchestration
```

## Example Use Cases

### Highway Traffic Analysis
- Automatically discover vehicle types (sedans, trucks, motorcycles)
- Track flow rates per vehicle type
- Monitor traffic patterns

### Retail Analytics
- Discover customer movement patterns
- Track foot traffic by detected categories
- Analyze flow through store sections

### Wildlife Monitoring
- Discover species from camera traps
- Track animal movements and counts
- Analyze migration patterns

## Output Format

### Flow Statistics

```json
{
  "elapsed_time": 120.5,
  "active_tracks": 12,
  "total_tracked": 156,
  "clusters": {
    "Cluster 0": {"active": 5, "flow_rate": "12.3/min", "avg_speed": "45.2"},
    "Cluster 1": {"active": 4, "flow_rate": "8.7/min", "avg_speed": "38.1"},
    "Cluster 2": {"active": 3, "flow_rate": "4.2/min", "avg_speed": "52.8"}
  }
}
```

### BigQuery Schema

| Column | Type | Description |
|--------|------|-------------|
| timestamp | TIMESTAMP | Event time |
| track_id | INTEGER | Unique track ID |
| cluster_id | INTEGER | Assigned cluster |
| cluster_label | STRING | Cluster name |
| bbox_* | INTEGER | Bounding box |
| center_* | INTEGER | Center position |
| velocity_* | FLOAT | Movement velocity |
| speed | FLOAT | Speed magnitude |

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 4GB+ RAM
- For YouTube streams: `yt-dlp` package

# GameCraft Streaming Clean Architecture

This directory contains a clean separation between frontend and backend for the GameCraft video streaming system.

## Architecture Overview

```
┌─────────────────┐    HTTP API    ┌──────────────────┐    File IPC    ┌─────────────────┐
│                 │   (port 8083)  │                  │  (trigger.txt) │                 │
│ Simple Frontend │◄──────────────►│ Streaming API    │◄──────────────►│ Streaming       │
│ (Web Interface) │                │ Server           │                │ Worker          │
│                 │                │ (Backend)        │                │ (Model)         │
└─────────────────┘                └──────────────────┘                └─────────────────┘
     Port 8085                           Port 8083                      Distributed GPUs
```

## Components

### 1. **Streaming Worker** (`streaming_worker.py`)
- **Purpose**: Distributed PyTorch worker that loads the video generation model
- **Technology**: PyTorch Distributed, torchrun
- **Communication**: File-based IPC (trigger.txt, result files)
- **GPUs**: Runs on 8 GPUs by default

### 2. **Streaming API Server** (`streaming_api_server.py`) 
- **Purpose**: Pure backend API server
- **Technology**: Flask REST API
- **Port**: 8083
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /start_stream` - Initialize video stream
  - `POST /control/<stream_id>` - Send WASD controls
  - `GET /video/<stream_id>` - MJPEG video stream
  - `POST /stop_stream/<stream_id>` - Stop stream

### 3. **Simple Frontend** (`simple_frontend.py`)
- **Purpose**: Clean frontend that serves HTML and connects to API
- **Technology**: Flask web server + HTML/CSS/JavaScript
- **Port**: 8085
- **Features**: 
  - Upload image
  - Set generation parameters
  - WASD controls
  - Real-time video streaming

## Usage

### Quick Start
```bash
# Start the clean architecture
./scripts/launch_streaming_clean.sh

# Open in browser
http://localhost:8085

# Stop all services (if Ctrl+C doesn't work)
./scripts/stop_streaming.sh
```

### Manual Start (for development)
```bash
# 1. Start the worker (loads model)
torchrun --nnodes=1 --nproc_per_node=8 --master_port 29607 hymm_sp/streaming/streaming_worker.py

# 2. Start the API server (backend)
STREAMING_PORT=8083 python hymm_sp/streaming/streaming_api_server.py

# 3. Start the frontend (web interface)
FRONTEND_PORT=8085 python hymm_sp/streaming/simple_frontend.py
```

## Benefits of This Architecture

### ✅ **Clean Separation**
- Frontend only handles UI
- Backend only handles API logic
- Worker only handles model inference

### ✅ **Multiple Frontend Support**
- Any frontend can use the API server
- Easy to create mobile apps, CLI tools, etc.
- API follows REST conventions

### ✅ **Development Friendly**
- Frontend can be developed independently
- API can be tested with curl/Postman
- Clear error boundaries

### ✅ **Scalable**
- Frontend and backend can run on different machines
- Multiple frontends can share one backend
- Easy to add load balancing

## API Examples

### Start a Stream
```bash
curl -X POST http://localhost:8083/start_stream \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "base64_image_data...",
    "prompt": "A beautiful landscape",
    "video_size": [352, 608],
    "cfg_scale": 1.0,
    "seed": 250161
  }'
```

### Send WASD Control
```bash
curl -X POST http://localhost:8083/control/stream_123 \
  -H "Content-Type: application/json" \
  -d '{
    "action": "w",
    "speed": 0.2
  }'
```

### Get Video Stream
```bash
# Open in browser or video player
http://localhost:8083/video/stream_123
```

## Compared to Previous Architecture

### Before (Messy)
- `streaming_app.py` had both UI and API endpoints (port 8085)
- `streaming_api_server.py` had duplicate API endpoints (port 8083)
- `streaming_interface.html` tried to connect to port 8083 but nothing ran there
- Confusing which component did what

### Now (Clean) 
- `simple_frontend.py` only serves UI (port 8085)
- `streaming_api_server.py` only serves API (port 8083)  
- `streaming_worker.py` only runs model inference
- Clear responsibilities and communication flow

## Troubleshooting

### Process Management Issues

**Problem**: Ctrl+C doesn't stop services properly
```bash
# Solution 1: Use the dedicated stop script
./scripts/stop_streaming.sh

# Solution 2: Kill by port (your method)
kill -9 $(lsof -t -i:8083)  # API server
kill -9 $(lsof -t -i:8085)  # Frontend

# Solution 3: Kill by process name
pkill -f streaming_api_server
pkill -f simple_frontend
pkill -f streaming_worker
```

**Problem**: "Cannot connect to API server"
- Check if API server is running: `curl http://localhost:8083/health`
- Ensure CORS is enabled (should see `Access-Control-Allow-Origin: *` header)
- Verify conda environment: `conda activate HYGameCraft`

**Problem**: Missing dependencies
```bash
conda activate HYGameCraft
pip install flask-cors
```

### Port Usage
- **8083**: Backend API server  
- **8085**: Frontend web interface
- **29607**: PyTorch distributed training master port

## Legacy Files

- `streaming_app.py` - Old monolithic app (UI + API combined)
- `streaming_interface.html` - Standalone HTML file  
- `launch_streaming.sh` - Old launch script

These are kept for reference but the clean architecture is recommended.

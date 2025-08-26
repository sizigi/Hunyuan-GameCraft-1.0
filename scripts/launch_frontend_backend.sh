#!/bin/bash

echo "🧹 Cleaning up any existing processes..."
pkill -f streaming_worker || true
pkill -f streaming_api_server || true  
pkill -f web_frontend || true
rm -f ./streaming_results/shutdown_worker.signal

mkdir -p ./streaming_results

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE="weights/stdmodels"
export CKPT_PATH="weights/gamecraft_models/mp_rank_00_model_states_distill.pt"
export API_PORT="8083"
export FRONTEND_PORT="8085"
NUM_GPUS=8

PROJECT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
WORKER_SCRIPT="$PROJECT_DIR/hymm_sp/streaming/streaming_worker.py"
API_SCRIPT="$PROJECT_DIR/hymm_sp/streaming/streaming_api_server.py"
FRONTEND_SCRIPT="$PROJECT_DIR/hymm_sp/streaming/web_frontend.py"

echo "🚀 Starting Frontend/Backend Architecture..."
echo "============================================"
echo "Frontend:    http://localhost:${FRONTEND_PORT}"
echo "Backend API: http://localhost:${API_PORT}"
echo "============================================"

# Function to check conda environment
check_conda_env() {
    if conda info --envs | grep -q "HYGameCraft"; then
        echo "✅ Found HYGameCraft conda environment"
        return 0
    else
        echo "❌ HYGameCraft conda environment not found"
        echo "Available environments:"
        conda info --envs
        return 1
    fi
}

# Check and setup
if ! check_conda_env; then
    echo "Please create the HYGameCraft environment first"
    exit 1
fi

echo "📦 Installing dependencies..."
conda run -n HYGameCraft pip install flask-cors requests 2>/dev/null || true

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate HYGameCraft

# Start API server and frontend in background first
echo ""
echo "🌐 Starting API Server (background)..."
env STREAMING_PORT=$API_PORT python "$API_SCRIPT" &
API_PID=$!
echo "✅ API Server PID: $API_PID"

echo ""
echo "🎨 Starting Frontend (background)..."
env FRONTEND_PORT=$FRONTEND_PORT API_URL="http://localhost:$API_PORT" python "$FRONTEND_SCRIPT" &
FRONTEND_PID=$!
echo "✅ Frontend PID: $FRONTEND_PID"

# Give them a moment to start
sleep 3

echo ""
echo "🎮 ============================================"
echo "🎮 Services Starting!"
echo "🎮 "
echo "🎮 🌐 Open in browser:"
echo "🎮   http://localhost:${FRONTEND_PORT}"
echo "🎮 "
echo "🎮 📊 Background Services:"
echo "🎮   - API PID: $API_PID (backend)"
echo "🎮   - Frontend PID: $FRONTEND_PID (web server)"
echo "🎮 ============================================"

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    # Send shutdown signal
    echo "shutdown" > ./streaming_results/shutdown_worker.signal 2>/dev/null || true
    echo "📤 Shutdown signal sent"
    
    # Kill processes by name (more reliable)
    pkill -f streaming_api_server 2>/dev/null || true
    pkill -f web_frontend 2>/dev/null || true
    pkill -f streaming_worker 2>/dev/null || true
    
    # Kill by PID as backup
    kill $FRONTEND_PID $API_PID 2>/dev/null || true
    
    # Wait briefly
    sleep 2
    
    # Force kill if needed
    kill -9 $FRONTEND_PID $API_PID 2>/dev/null || true
    pkill -9 -f streaming_worker 2>/dev/null || true
    
    echo "✅ Services stopped"
    exit 0
}

# Set up signal handling
trap cleanup INT TERM

# Now start worker in FOREGROUND so we can see all the logs!
echo ""
echo "🔧 Starting Streaming Worker (FOREGROUND - you'll see all logs)..."
echo "📄 Model loading progress will be shown below:"
echo "============================================"

# Run worker in foreground - this will show all the model loading logs!
torchrun \
--nnodes=1 \
--nproc_per_node=$NUM_GPUS \
--master_port 29607 \
"$WORKER_SCRIPT"

# If we get here, worker exited - cleanup
cleanup

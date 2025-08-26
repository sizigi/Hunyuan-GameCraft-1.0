#!/bin/bash

mkdir -p ./streaming_results
pkill -f streaming_worker || true
pkill -f streaming_api_server || true
echo "🧹 Cleaning up potential stale shutdown signal file..."
rm -f ./streaming_results/shutdown_worker.signal

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE="weights/stdmodels"
export CKPT_PATH="weights/gamecraft_models/mp_rank_00_model_states_distill.pt"
export STREAMING_PORT="8085"  # Port for streaming API server (changed to avoid conflict)
NUM_GPUS=8

PROJECT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
WORKER_SCRIPT="$PROJECT_DIR/hymm_sp/streaming/streaming_worker.py"
APP_SCRIPT="$PROJECT_DIR/hymm_sp/streaming/streaming_app.py"

echo "🚀 Starting Streaming Inference Worker..."
echo "============================================"
echo "Starting model loading on $NUM_GPUS GPUs..."
echo "============================================"

# Run worker in foreground to see all output
torchrun \
--nnodes=1 \
--nproc_per_node=$NUM_GPUS \
--master_port 29607 \
"$WORKER_SCRIPT" 2>&1 | tee streaming_results/worker.log &

WORKER_PID=$!
echo "✅ Streaming Worker started with PID $WORKER_PID"

echo "🌐 Starting Streaming App..."
python3 "$APP_SCRIPT" > streaming_results/app.log 2>&1 &
APP_PID=$!

echo "⏳ Waiting for Worker to load model and initialize..."
sleep 40

echo "🔍 Checking Streaming App health..."

HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_INTERVAL=15
HEALTHY=false

for ((i=1; i<=HEALTH_CHECK_RETRIES; i++)); do
  echo "---- Health Check Attempt $i ----"
  RESPONSE=$(curl -s --noproxy localhost http://localhost:${STREAMING_PORT}/api/health)
  CURL_EXIT_CODE=$?
  
  echo "Curl Exit Code: $CURL_EXIT_CODE"
  echo "Response Body: $(echo "$RESPONSE" | head -c 500)"
  
  if [ $CURL_EXIT_CODE -eq 0 ] && echo "$RESPONSE" | grep -q '"status"[[:space:]]*:[[:space:]]*"healthy"'; then
    echo "✅ Streaming App is healthy."
    HEALTHY=true
    break
  else
    if [ $CURL_EXIT_CODE -ne 0 ]; then
      echo "⚠️  Curl command failed."
    else
      echo "⚠️  App responded, but status is not healthy."
    fi
  fi
  echo "--------------------"
  
  if [ $i -lt $HEALTH_CHECK_RETRIES ]; then
    echo "⏳ Waiting for App to be healthy... (attempt $i/$HEALTH_CHECK_RETRIES)"
    sleep $HEALTH_CHECK_INTERVAL
  fi
done

if [ "$HEALTHY" = false ]; then
  echo "❌ Streaming App failed to become healthy. Check streaming_results/app.log and streaming_results/worker.log"
  kill $WORKER_PID $APP_PID 2>/dev/null || true
  wait $WORKER_PID $APP_PID 2>/dev/null || true
  exit 1
fi

echo "✅ Streaming App is healthy."
echo ""
echo "🎮 ============================================"
echo "🎮 Video Streaming Server is ready!"
echo "🎮 "
echo "🎮 Open in your browser:"
echo "🎮   http://localhost:${STREAMING_PORT}  (port 8085)"
echo "🎮 "
echo "🎮 API Endpoints:"
echo "🎮   - Web Interface: http://localhost:${STREAMING_PORT}/"
echo "🎮   - Health Check: http://localhost:${STREAMING_PORT}/api/health"
echo "🎮   - Start Stream: POST http://localhost:${STREAMING_PORT}/api/start_stream"
echo "🎮   - Control: POST http://localhost:${STREAMING_PORT}/api/control/<stream_id>"
echo "🎮   - Video Stream: http://localhost:${STREAMING_PORT}/api/video/<stream_id>"
echo "🎮 ============================================"
echo ""
echo "Press Ctrl+C to stop all services..."

# Function to handle cleanup
cleanup() {
  echo ""
  echo "🛑 Shutting down streaming services..."
  echo "shutdown" > ./streaming_results/shutdown_worker.signal
  echo "📤 Shutdown signal sent to worker."
  echo "⏳ Waiting for services to shut down..."
  
  # Try graceful shutdown first
  timeout 30 tail --pid=$WORKER_PID -f /dev/null 2>/dev/null || true
  timeout 10 tail --pid=$APP_PID -f /dev/null 2>/dev/null || true
  
  # Force kill if still running
  kill $WORKER_PID $APP_PID 2>/dev/null || true
  wait $WORKER_PID $APP_PID 2>/dev/null || true
  
  echo "🔚 All streaming services shut down."
}

# Set up trap for cleanup
trap cleanup INT TERM

# Wait for background processes
wait $WORKER_PID $APP_PID
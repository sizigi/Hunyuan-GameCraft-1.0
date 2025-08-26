# streaming_api_server.py
from flask import Flask, Response, request, jsonify, send_file
import os
import json
import time
import base64
import cv2
import numpy as np
from loguru import logger
import signal
import sys
import traceback
from io import BytesIO
from PIL import Image
import threading
import queue
from pathlib import Path

app = Flask(__name__)

# ========== Configuration ==========
TRIGGER_FILE = "./streaming_results/trigger.txt"
RESULT_DIR = "./streaming_results"
INPUT_IMAGES_DIR = os.path.join(RESULT_DIR, "input")
SHUTDOWN_FILE = "./streaming_results/shutdown_worker.signal"
MAX_WAIT_TIME = 30000
CHECK_INTERVAL = 0.1
STREAMING_PORT = int(os.getenv("STREAMING_PORT", 8083))

# Video streaming configuration
VIDEO_FPS = 30
CHUNK_SIZE = 9  # frames per chunk (matching sample-n-frames from launch-distill.sh)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)

# Global state for streaming
active_streams = {}
stream_lock = threading.Lock()

class VideoStream:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.frame_queue = queue.Queue(maxsize=100)
        self.is_active = True
        self.current_action = None
        self.generation_params = {}
        
    def add_frames(self, frames):
        """Add frames to the stream queue"""
        for frame in frames:
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                
    def get_frame(self):
        """Get next frame from queue"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return None
            
    def stop(self):
        """Stop the stream"""
        self.is_active = False

@app.route('/health', methods=['GET'])
def health():
    worker_ready = os.path.exists(RESULT_DIR) and os.access(RESULT_DIR, os.W_OK)
    return jsonify({
        "status": "healthy" if worker_ready else "unhealthy",
        "worker_ready": worker_ready,
        "active_streams": len(active_streams)
    })

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Initialize a new video stream with base parameters"""
    try:
        data = request.get_json() or {}
        
        # Extract parameters matching launch-distill.sh defaults
        params = {
            "image_path": data.get("image_path"),
            "image_base64": data.get("image_base64"),
            "prompt": data.get("prompt", "Realistic, High-quality."),
            "add_neg_prompt": data.get("add_neg_prompt", 
                "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border."),
            "video_size": data.get("video_size", [352, 608]),
            "cfg_scale": data.get("cfg_scale", 1.0),
            "seed": data.get("seed", 250161),
            "infer_steps": data.get("infer_steps", 8),
            "use_fp8": data.get("use_fp8", True),
            "flow_shift_eval_video": data.get("flow_shift_eval_video", 5.0),
            "sample_n_frames": data.get("sample_n_frames", 9)
        }
        
        # Save image if base64 provided
        if params["image_base64"]:
            timestamp = int(time.time() * 1000)
            image_path = save_base64_image(
                params["image_base64"], 
                INPUT_IMAGES_DIR, 
                f"stream_{timestamp}"
            )
            if image_path:
                params["image_path"] = image_path
            del params["image_base64"]
        
        # Generate stream ID
        stream_id = f"stream_{int(time.time() * 1000)}"
        
        # Create new stream
        with stream_lock:
            stream = VideoStream(stream_id)
            stream.generation_params = params
            active_streams[stream_id] = stream
        
        logger.info(f"Started new stream: {stream_id}")
        
        # Trigger initial generation
        trigger_generation(stream_id, "w", 0.2)  # Start with forward movement
        
        return jsonify({
            "stream_id": stream_id,
            "status": "started",
            "params": params
        })
        
    except Exception as e:
        logger.error(f"Failed to start stream: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/control/<stream_id>', methods=['POST'])
def control_stream(stream_id):
    """Send WASD control to active stream"""
    try:
        data = request.get_json() or {}
        action = data.get("action", "w").lower()
        speed = data.get("speed", 0.2)
        
        # Validate action
        valid_actions = ["w", "a", "s", "d"]
        if action not in valid_actions:
            return jsonify({"error": f"Invalid action. Must be one of {valid_actions}"}), 400
        
        with stream_lock:
            if stream_id not in active_streams:
                return jsonify({"error": "Stream not found"}), 404
            
            stream = active_streams[stream_id]
            stream.current_action = action
        
        # Trigger new generation with action
        trigger_generation(stream_id, action, speed)
        
        return jsonify({
            "stream_id": stream_id,
            "action": action,
            "speed": speed,
            "status": "action_sent"
        })
        
    except Exception as e:
        logger.error(f"Failed to control stream: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/video/<stream_id>')
def stream_video(stream_id):
    """Stream video chunks in MJPEG format"""
    
    def generate():
        with stream_lock:
            if stream_id not in active_streams:
                return
            stream = active_streams[stream_id]
        
        logger.info(f"Starting video stream for {stream_id}")
        
        while stream.is_active:
            # Wait for frames from worker
            frames = []
            deadline = time.time() + 1.0  # Collect frames for up to 1 second
            
            while time.time() < deadline and len(frames) < CHUNK_SIZE:
                frame = stream.get_frame()
                if frame is not None:
                    frames.append(frame)
                else:
                    time.sleep(0.01)
            
            # Send collected frames as MJPEG
            for frame in frames:
                try:
                    # Encode frame as JPEG
                    if isinstance(frame, np.ndarray):
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        jpg = buffer.tobytes()
                    elif isinstance(frame, bytes):
                        jpg = frame
                    else:
                        continue
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
                except Exception as e:
                    logger.error(f"Error encoding frame: {e}")
            
            # If no frames available, wait a bit
            if not frames:
                time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream/<stream_id>', methods=['POST'])
def stop_stream(stream_id):
    """Stop an active stream"""
    try:
        with stream_lock:
            if stream_id not in active_streams:
                return jsonify({"error": "Stream not found"}), 404
            
            stream = active_streams[stream_id]
            stream.stop()
            del active_streams[stream_id]
        
        logger.info(f"Stopped stream: {stream_id}")
        return jsonify({"status": "stopped", "stream_id": stream_id})
        
    except Exception as e:
        logger.error(f"Failed to stop stream: {e}")
        return jsonify({"error": str(e)}), 500

def trigger_generation(stream_id, action, speed):
    """Trigger video generation with worker"""
    try:
        with stream_lock:
            if stream_id not in active_streams:
                return
            stream = active_streams[stream_id]
        
        # Prepare trigger data
        trigger_data = {
            "stream_id": stream_id,
            "custom_params": {
                **stream.generation_params,
                "action_list": [action],
                "action_speed_list": [speed],
                "streaming": True
            }
        }
        
        # Write trigger file
        trigger_tmp = TRIGGER_FILE + ".tmp"
        with open(trigger_tmp, 'w') as f:
            json.dump(trigger_data, f, indent=2, ensure_ascii=False)
        os.replace(trigger_tmp, TRIGGER_FILE)
        
        logger.info(f"Triggered generation for {stream_id} with action {action}")
        
        # Start thread to wait for results
        threading.Thread(
            target=wait_for_results,
            args=(stream_id,),
            daemon=True
        ).start()
        
    except Exception as e:
        logger.error(f"Failed to trigger generation: {e}")
        logger.error(traceback.format_exc())

def wait_for_results(stream_id):
    """Wait for and process generation results"""
    result_file = os.path.join(RESULT_DIR, f"result_{stream_id}.json")
    start_time = time.time()
    
    while time.time() - start_time < MAX_WAIT_TIME:
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                # Process video frames
                if "video_path" in result:
                    add_video_to_stream(stream_id, result["video_path"])
                elif "frames" in result:
                    # Direct frame data
                    with stream_lock:
                        if stream_id in active_streams:
                            stream = active_streams[stream_id]
                            stream.add_frames(result["frames"])
                
                # Clean up
                os.remove(result_file)
                logger.info(f"Processed result for {stream_id}")
                return
                
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                if os.path.exists(result_file):
                    try:
                        os.remove(result_file)
                    except:
                        pass
        
        time.sleep(CHECK_INTERVAL)
    
    logger.warning(f"Timeout waiting for results for {stream_id}")

def add_video_to_stream(stream_id, video_path):
    """Extract frames from video and add to stream"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        with stream_lock:
            if stream_id in active_streams:
                stream = active_streams[stream_id]
                stream.add_frames(frames)
                logger.info(f"Added {len(frames)} frames to stream {stream_id}")
        
    except Exception as e:
        logger.error(f"Error adding video to stream: {e}")

def save_base64_image(base64_string, save_dir, prefix="input"):
    """Save base64 image to file"""
    if not base64_string:
        return None
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        filename = f"{prefix}.png"
        file_path = os.path.join(save_dir, filename)
        image.save(file_path, "PNG")
        logger.info(f"Saved image to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return None

def signal_handler(sig, frame):
    """Handle shutdown signal"""
    logger.info("Received shutdown signal...")
    
    # Stop all active streams
    with stream_lock:
        for stream_id, stream in active_streams.items():
            stream.stop()
        active_streams.clear()
    
    # Notify worker
    try:
        with open(SHUTDOWN_FILE, 'w') as f:
            f.write("shutdown")
        logger.info("Shutdown signal sent to worker.")
    except Exception as e:
        logger.error(f"Failed to send shutdown signal: {e}")
    
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        logger.info(f"🚀 Starting Streaming API server on port {STREAMING_PORT}...")
        app.run(host="0.0.0.0", port=STREAMING_PORT, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
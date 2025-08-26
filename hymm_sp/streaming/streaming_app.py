# streaming_app.py
from flask import Flask, Response, request, jsonify, render_template_string, send_from_directory
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
STREAMING_PORT = int(os.getenv("STREAMING_PORT", 8085))

# Video streaming configuration
VIDEO_FPS = 30
CHUNK_SIZE = 9  # frames per chunk

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)

# Global state for streaming
active_streams = {}
stream_lock = threading.Lock()

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GameCraft Video Streaming</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 1400px;
            width: 100%;
            padding: 30px;
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
        }
        
        .control-panel {
            background: #f7f7f7;
            padding: 20px;
            border-radius: 10px;
        }
        
        .section {
            margin-bottom: 25px;
        }
        
        .section h2 {
            color: #555;
            font-size: 1.2em;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            color: #666;
            font-weight: 500;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        
        input[type="text"],
        input[type="number"],
        textarea,
        select {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        
        textarea {
            resize: vertical;
            min-height: 60px;
        }
        
        .file-upload {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        
        .file-upload input[type="file"] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .file-upload-label {
            display: block;
            padding: 10px;
            background: #667eea;
            color: white;
            text-align: center;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .file-upload-label:hover {
            background: #5a67d8;
        }
        
        .button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
            width: 100%;
        }
        
        .button:hover {
            background: #5a67d8;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .button.stop {
            background: #e53e3e;
            margin-top: 10px;
        }
        
        .button.stop:hover {
            background: #c53030;
        }
        
        .wasd-controls {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            max-width: 200px;
            margin: 20px auto;
        }
        
        .wasd-btn {
            padding: 20px;
            background: #4a5568;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 20px;
            font-weight: bold;
            transition: all 0.2s;
        }
        
        .wasd-btn:hover {
            background: #2d3748;
            transform: scale(1.05);
        }
        
        .wasd-btn:active {
            transform: scale(0.95);
            background: #667eea;
        }
        
        .wasd-btn.w { grid-column: 2; }
        .wasd-btn.a { grid-column: 1; }
        .wasd-btn.s { grid-column: 2; }
        .wasd-btn.d { grid-column: 3; }
        
        .video-container {
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            min-height: 500px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        #videoStream {
            width: 100%;
            height: auto;
            display: none;
        }
        
        .placeholder {
            color: #888;
            font-size: 1.2em;
            text-align: center;
        }
        
        .status {
            margin-top: 15px;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .status.success {
            background: #c6f6d5;
            color: #22543d;
        }
        
        .status.error {
            background: #fed7d7;
            color: #742a2a;
        }
        
        .status.info {
            background: #bee3f8;
            color: #2c5282;
        }
        
        .preview-image {
            max-width: 100%;
            margin-top: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        
        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .speed-control input[type="range"] {
            flex: 1;
        }
        
        .speed-value {
            min-width: 40px;
            text-align: center;
            font-weight: bold;
            color: #667eea;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 GameCraft Video Streaming</h1>
        
        <div class="main-content">
            <div class="control-panel">
                <div class="section">
                    <h2>📸 Input Image</h2>
                    <div class="form-group">
                        <div class="file-upload">
                            <input type="file" id="imageUpload" accept="image/*">
                            <label for="imageUpload" class="file-upload-label">Choose Image</label>
                        </div>
                        <img id="imagePreview" class="preview-image" style="display: none;">
                    </div>
                </div>
                
                <div class="section">
                    <h2>⚙️ Generation Parameters</h2>
                    
                    <div class="form-group">
                        <label for="prompt">Prompt:</label>
                        <textarea id="prompt" placeholder="Describe the scene...">A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky.</textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="videoSize">Video Size:</label>
                        <select id="videoSize">
                            <option value="352,608" selected>352x608 (Fast)</option>
                            <option value="704,1216">704x1216 (HD)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="cfgScale">CFG Scale:</label>
                        <input type="number" id="cfgScale" value="1.0" min="0.1" max="10" step="0.1">
                    </div>
                    
                    <div class="form-group">
                        <label for="seed">Seed:</label>
                        <input type="number" id="seed" value="250161" min="0">
                    </div>
                    
                    <div class="form-group">
                        <label for="inferSteps">Inference Steps:</label>
                        <input type="number" id="inferSteps" value="8" min="1" max="50">
                    </div>
                    
                    <div class="form-group">
                        <label for="framesPerChunk">Frames per Chunk:</label>
                        <input type="number" id="framesPerChunk" value="9" min="1" max="33">
                    </div>
                    
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="useFp8" checked>
                            Use FP8 (Faster inference)
                        </label>
                    </div>
                </div>
                
                <button id="startBtn" class="button">🚀 Start Streaming</button>
                <button id="stopBtn" class="button stop" style="display: none;">⏹ Stop Streaming</button>
                
                <div id="status" class="status" style="display: none;"></div>
            </div>
            
            <div class="video-section">
                <div class="video-container">
                    <img id="videoStream" alt="Video Stream">
                    <div class="placeholder">Video stream will appear here</div>
                </div>
                
                <div class="section" id="controlsSection" style="display: none;">
                    <h2>🎮 WASD Controls</h2>
                    
                    <div class="speed-control">
                        <label for="actionSpeed">Movement Speed:</label>
                        <input type="range" id="actionSpeed" min="0.1" max="1.0" step="0.1" value="0.2">
                        <span class="speed-value">0.2</span>
                    </div>
                    
                    <div class="wasd-controls">
                        <button class="wasd-btn w" data-action="w">W</button>
                        <button class="wasd-btn a" data-action="a">A</button>
                        <button class="wasd-btn s" data-action="s">S</button>
                        <button class="wasd-btn d" data-action="d">D</button>
                    </div>
                    
                    <p style="text-align: center; color: #666; margin-top: 10px;">
                        Use WASD keys or click buttons to control movement
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentStreamId = null;
        let isStreaming = false;
        
        // DOM elements
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const videoStream = document.getElementById('videoStream');
        const statusDiv = document.getElementById('status');
        const imageUpload = document.getElementById('imageUpload');
        const imagePreview = document.getElementById('imagePreview');
        const controlsSection = document.getElementById('controlsSection');
        const speedSlider = document.getElementById('actionSpeed');
        const speedValue = document.querySelector('.speed-value');
        const placeholder = document.querySelector('.placeholder');
        
        // Image upload handling
        imageUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Speed slider
        speedSlider.addEventListener('input', (e) => {
            speedValue.textContent = e.target.value;
        });
        
        // Convert image to base64
        async function imageToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const base64 = reader.result.split(',')[1];
                    resolve(base64);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        }
        
        // Start streaming
        startBtn.addEventListener('click', async () => {
            if (!imageUpload.files[0]) {
                showStatus('Please select an image first', 'error');
                return;
            }
            
            startBtn.disabled = true;
            showStatus('Starting stream...', 'info');
            
            try {
                const imageBase64 = await imageToBase64(imageUpload.files[0]);
                const videoSize = document.getElementById('videoSize').value.split(',').map(Number);
                
                const params = {
                    image_base64: imageBase64,
                    prompt: document.getElementById('prompt').value,
                    video_size: videoSize,
                    cfg_scale: parseFloat(document.getElementById('cfgScale').value),
                    seed: parseInt(document.getElementById('seed').value),
                    infer_steps: parseInt(document.getElementById('inferSteps').value),
                    sample_n_frames: parseInt(document.getElementById('framesPerChunk').value),
                    use_fp8: document.getElementById('useFp8').checked
                };
                
                const response = await fetch('/api/start_stream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(params)
                });
                
                if (!response.ok) {
                    throw new Error(`Failed to start stream: ${response.statusText}`);
                }
                
                const data = await response.json();
                currentStreamId = data.stream_id;
                isStreaming = true;
                
                // Update UI
                startBtn.style.display = 'none';
                stopBtn.style.display = 'block';
                controlsSection.style.display = 'block';
                placeholder.style.display = 'none';
                videoStream.style.display = 'block';
                
                // Start video stream
                videoStream.src = `/api/video/${currentStreamId}`;
                
                showStatus('Stream started successfully!', 'success');
                
            } catch (error) {
                console.error('Error starting stream:', error);
                showStatus(`Error: ${error.message}`, 'error');
                startBtn.disabled = false;
            }
        });
        
        // Stop streaming
        stopBtn.addEventListener('click', async () => {
            if (!currentStreamId) return;
            
            try {
                await fetch(`/api/stop_stream/${currentStreamId}`, {
                    method: 'POST'
                });
                
                isStreaming = false;
                currentStreamId = null;
                
                // Reset UI
                startBtn.style.display = 'block';
                startBtn.disabled = false;
                stopBtn.style.display = 'none';
                controlsSection.style.display = 'none';
                videoStream.style.display = 'none';
                videoStream.src = '';
                placeholder.style.display = 'flex';
                
                showStatus('Stream stopped', 'info');
                
            } catch (error) {
                console.error('Error stopping stream:', error);
                showStatus(`Error: ${error.message}`, 'error');
            }
        });
        
        // WASD control handling
        async function sendAction(action) {
            if (!isStreaming || !currentStreamId) return;
            
            console.log(`🎮 CLIENT: Sending action '${action}' with speed ${parseFloat(speedSlider.value)}`);
            
            try {
                const payload = {
                    action: action,
                    speed: parseFloat(speedSlider.value)
                };
                console.log('📤 CLIENT: Payload:', payload);
                
                const response = await fetch(`/api/control/${currentStreamId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                
                if (!response.ok) {
                    throw new Error(`Failed to send action: ${response.statusText}`);
                }
                
                const result = await response.json();
                console.log('📥 CLIENT: Response:', result);
                
            } catch (error) {
                console.error('Error sending action:', error);
                showStatus(`Error: ${error.message}`, 'error');
            }
        }
        
        // Button controls
        document.querySelectorAll('.wasd-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                console.log(`🖱️ CLIENT: Button clicked - action='${action}'`);
                sendAction(action);
            });
        });
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (!isStreaming) return;
            
            const key = e.key.toLowerCase();
            console.log(`⌨️ CLIENT: Key pressed - '${key}'`);
            
            if (['w', 'a', 's', 'd'].includes(key)) {
                e.preventDefault();
                console.log(`⌨️ CLIENT: Valid WASD key - sending action '${key}'`);
                sendAction(key);
                
                // Visual feedback
                const btn = document.querySelector(`.wasd-btn[data-action="${key}"]`);
                if (btn) {
                    btn.style.background = '#667eea';
                    setTimeout(() => {
                        btn.style.background = '#4a5568';
                    }, 200);
                }
            }
        });
        
        // Status display
        function showStatus(message, type) {
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            statusDiv.style.display = 'block';
            
            if (type === 'success' || type === 'info') {
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 3000);
            }
        }
        
        // Check API health on load
        async function checkHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                
                if (data.status === 'healthy') {
                    showStatus('API server is ready', 'success');
                } else {
                    showStatus('API server is not ready. Please check the worker.', 'error');
                    startBtn.disabled = true;
                }
            } catch (error) {
                showStatus('Cannot connect to API server', 'error');
                startBtn.disabled = true;
            }
        }
        
        // Initialize
        checkHealth();
    </script>
</body>
</html>
'''

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

# Web interface route
@app.route('/')
def index():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

# API endpoints
@app.route('/api/health', methods=['GET'])
def health():
    worker_ready = os.path.exists(RESULT_DIR) and os.access(RESULT_DIR, os.W_OK)
    return jsonify({
        "status": "healthy" if worker_ready else "unhealthy",
        "worker_ready": worker_ready,
        "active_streams": len(active_streams)
    })

@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    """Initialize a new video stream"""
    try:
        data = request.get_json() or {}
        
        # Extract parameters
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
        trigger_generation(stream_id, "w", 0.2)
        
        return jsonify({
            "stream_id": stream_id,
            "status": "started",
            "params": params
        })
        
    except Exception as e:
        logger.error(f"Failed to start stream: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/control/<stream_id>', methods=['POST'])
def control_stream(stream_id):
    """Send WASD control to active stream"""
    try:
        data = request.get_json() or {}
        action = data.get("action", "w").lower()
        speed = data.get("speed", 0.2)
        
        logger.info(f"🎮 CONTROL REQUEST: stream={stream_id}, action={action}, speed={speed}")
        
        # Validate action
        valid_actions = ["w", "a", "s", "d"]
        if action not in valid_actions:
            return jsonify({"error": f"Invalid action. Must be one of {valid_actions}"}), 400
        
        with stream_lock:
            if stream_id not in active_streams:
                return jsonify({"error": "Stream not found"}), 404
            
            stream = active_streams[stream_id]
            stream.current_action = action
            logger.info(f"📝 Updated stream {stream_id} current_action to: {action}")
        
        # Trigger new generation with action
        logger.info(f"🚀 Triggering generation with action: {action}")
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

@app.route('/api/video/<stream_id>')
def stream_video(stream_id):
    """Stream video chunks in MJPEG format"""
    
    def generate():
        with stream_lock:
            if stream_id not in active_streams:
                logger.error(f"Stream {stream_id} not found in active streams")
                return
            stream = active_streams[stream_id]
        
        logger.info(f"Starting video stream for {stream_id}")
        frames_sent = 0
        
        while stream.is_active:
            # Wait for frames from worker
            frames = []
            deadline = time.time() + 1.0
            
            while time.time() < deadline and len(frames) < CHUNK_SIZE:
                frame = stream.get_frame()
                if frame is not None:
                    frames.append(frame)
                else:
                    time.sleep(0.01)
            
            if len(frames) > 0:
                logger.info(f"📺 Sending {len(frames)} frames for stream {stream_id}, total sent: {frames_sent}")
            
            # Send collected frames as MJPEG
            for frame in frames:
                try:
                    # Encode frame as JPEG
                    if isinstance(frame, np.ndarray):
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        jpg = buffer.tobytes()
                        frames_sent += 1
                    elif isinstance(frame, bytes):
                        jpg = frame
                    else:
                        continue
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
                except Exception as e:
                    logger.error(f"Error encoding frame: {e}")
            
            # If no frames available, wait
            if not frames:
                time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stop_stream/<stream_id>', methods=['POST'])
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
        logger.info(f"🎯 PREPARING TRIGGER: action={action}, speed={speed}")
        trigger_data = {
            "stream_id": stream_id,
            "custom_params": {
                **stream.generation_params,
                "action_list": [action],
                "action_speed_list": [speed],
                "streaming": True
            }
        }
        
        # Log the exact action being sent
        logger.info(f"📤 SENDING ACTION TO WORKER: {trigger_data['custom_params']['action_list']}")
        
        # Write trigger file
        trigger_tmp = TRIGGER_FILE + ".tmp"
        with open(trigger_tmp, 'w') as f:
            json.dump(trigger_data, f, indent=2, ensure_ascii=False)
        os.replace(trigger_tmp, TRIGGER_FILE)
        
        logger.info(f"✅ Wrote trigger file: {TRIGGER_FILE}")
        logger.info(f"📋 Trigger content: {json.dumps(trigger_data, indent=2)[:500]}...")
        logger.info(f"🎬 Triggered generation for {stream_id} with action {action}")
        
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
    
    logger.info(f"⏳ Waiting for result file: {result_file}")
    check_count = 0
    
    while time.time() - start_time < MAX_WAIT_TIME:
        check_count += 1
        if check_count % 50 == 0:  # Log every 5 seconds
            elapsed = time.time() - start_time
            logger.debug(f"Still waiting for {stream_id} - {elapsed:.1f}s elapsed")
        
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
        logger.info(f"📼 Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"📊 Video info: {frame_count} frames at {fps} FPS")
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        logger.info(f"✅ Read {len(frames)} frames from video")
        
        with stream_lock:
            if stream_id in active_streams:
                stream = active_streams[stream_id]
                stream.add_frames(frames)
                logger.info(f"🎬 Added {len(frames)} frames to stream {stream_id}")
                logger.info(f"📦 Queue size: {stream.frame_queue.qsize()}")
            else:
                logger.error(f"❌ Stream {stream_id} not found in active streams")
        
    except Exception as e:
        logger.error(f"Error adding video to stream: {e}")
        logger.error(traceback.format_exc())

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
        logger.info(f"🚀 Starting Streaming App on port {STREAMING_PORT}...")
        logger.info(f"🌐 Open http://localhost:{STREAMING_PORT} in your browser")
        app.run(host="0.0.0.0", port=STREAMING_PORT, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
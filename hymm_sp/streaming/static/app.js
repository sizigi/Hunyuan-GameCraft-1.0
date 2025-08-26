// Frontend uses proxied API calls (CORS-free with fixed MJPEG proxy)
const API_URL = 'http://100.94.144.57:8083';
let currentStreamId = null;
let isStreaming = false;
let videoStreamStarted = false;

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
        // Stop any existing stream first to prevent conflicts
        if (currentStreamId && isStreaming) {
            console.log('🛑 Stopping existing stream before starting new one');
            try {
                await fetch(`${API_URL}/stop_stream/${currentStreamId}`, {
                    method: 'POST'
                });
            } catch (e) {
                console.warn('⚠️ Failed to stop existing stream:', e);
            }
        }
        
        // Reset state
        currentStreamId = null;
        isStreaming = false;
        videoStreamStarted = false;
        
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
        
        console.log('📤 Starting new stream with params:', params);
        
        const response = await fetch(`${API_URL}/start_stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to start stream: ${response.statusText} - ${errorText}`);
        }
        
        const data = await response.json();
        currentStreamId = data.stream_id;
        isStreaming = true;
        
        console.log('✅ New stream started:', data);
        
        // Update UI
        startBtn.style.display = 'none';
        stopBtn.style.display = 'block';
        controlsSection.style.display = 'block';
        placeholder.style.display = 'none';
        
        // Show uploaded image as initial frame
        videoStream.style.display = 'block';
        videoStream.src = imagePreview.src;
        
        showStatus('Stream ready! Use WASD controls to start movement.', 'success');
        
    } catch (error) {
        console.error('❌ Error starting stream:', error);
        showStatus(`Error: ${error.message}`, 'error');
        startBtn.disabled = false;
    }
});

// Stop streaming
stopBtn.addEventListener('click', async () => {
    if (!currentStreamId) return;

    try {
        await fetch(`${API_URL}/stop_stream/${currentStreamId}`, {
            method: 'POST'
        });

        isStreaming = false;
        currentStreamId = null;
        videoStreamStarted = false;

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
    if (!isStreaming || !currentStreamId) {
        console.warn('⚠️ CLIENT: Cannot send action - not streaming or no stream ID');
        return;
    }

        // Switch to video stream on first action
    if (!videoStreamStarted) {
        console.log(`🎬 CLIENT: Starting video stream for: ${currentStreamId}`);
        const videoUrl = `${API_URL}/video/${currentStreamId}`;
        console.log(`📺 CLIENT: Video URL: ${videoUrl}`);
        
        videoStreamStarted = true;
        showStatus('Video generation started... Please wait (~30-40 seconds)', 'info');
        
        // Add event listeners to debug video loading
        videoStream.onloadstart = () => {
            console.log('📺 VIDEO: Load started');
            showStatus('Video stream connecting...', 'info');
        };
        videoStream.onprogress = () => console.log('📺 VIDEO: Progress');
        videoStream.onloadeddata = () => {
            console.log('📺 VIDEO: Data loaded');
            showStatus('Video stream ready!', 'success');
        };
        videoStream.onerror = (e) => {
            console.error('❌ VIDEO ERROR:', e);
            showStatus('Video stream error - retrying...', 'error');
            // Retry after error
            setTimeout(() => {
                console.log('🔄 VIDEO: Retrying after error');
                videoStream.src = videoUrl + '?retry=' + Date.now();
            }, 2000);
        };
        videoStream.onload = () => {
            console.log('✅ VIDEO: Loaded successfully');
            showStatus('Video streaming active!', 'success');
        };
        
        // Start the video stream immediately (it will wait for frames)
        videoStream.src = videoUrl;
        
        // Add retry mechanism for robustness
        let retryCount = 0;
        const maxRetries = 3;
        
        const retryConnection = () => {
            if (retryCount < maxRetries) {
                retryCount++;
                console.log(`🔄 VIDEO: Retry attempt ${retryCount}/${maxRetries}`);
                showStatus(`Connecting to video stream (attempt ${retryCount}/${maxRetries})...`, 'info');
                videoStream.src = videoUrl + '?retry=' + retryCount + '&t=' + Date.now();
            } else {
                showStatus('Video stream connection failed after multiple attempts', 'error');
            }
        };
        
        // Retry if no data after 45 seconds (video generation usually takes ~40s)
        setTimeout(() => {
            if (videoStream.readyState === 0) { // HAVE_NOTHING
                console.log('⏰ VIDEO: No data after 45s, retrying...');
                retryConnection();
            }
        }, 45000);
    }

    console.log(`🎮 CLIENT: Sending action '${action}' with speed ${parseFloat(speedSlider.value)}`);

    try {
        const payload = {
            action: action,
            speed: parseFloat(speedSlider.value)
        };
        console.log('📤 CLIENT: Payload:', payload);
        console.log(`📡 CLIENT: Sending to: ${API_URL}/control/${currentStreamId}`);

        const response = await fetch(`${API_URL}/control/${currentStreamId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        console.log(`📥 CLIENT: Response status: ${response.status}`);

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`❌ CLIENT: Bad response: ${response.status} - ${errorText}`);
            throw new Error(`Failed to send action: ${response.statusText} - ${errorText}`);
        }

        const result = await response.json();
        console.log('✅ CLIENT: Success response:', result);

        if (result.status === 'action_sent') {
            showStatus(`Action '${action}' sent successfully`, 'success');
        }

    } catch (error) {
        console.error('❌ CLIENT: Error sending action:', error);
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
    if (['w', 'a', 's', 'd'].includes(key)) {
        e.preventDefault();
        console.log(`⌨️ CLIENT: Key pressed - '${key}'`);
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
        console.log('🔍 Checking API health...');
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();

        console.log('📊 API Health:', data);

        if (data.status === 'healthy') {
            showStatus('API server is ready', 'success');
        } else {
            showStatus('API server is not ready. Please check the worker.', 'error');
            startBtn.disabled = true;
        }
    } catch (error) {
        console.error('❌ Cannot connect to API server:', error);
        showStatus('Cannot connect to API server', 'error');
        startBtn.disabled = true;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Frontend loaded, checking health...');
    checkHealth();
});

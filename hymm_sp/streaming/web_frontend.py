#!/usr/bin/env python3
"""
Web Frontend Server - Clean separation with actual HTML/CSS/JS files
NO HTML IN PYTHON! Serves static files and proxies API calls.
"""

from flask import Flask, send_file, send_from_directory, jsonify, request, Response
import requests
import os
from loguru import logger

app = Flask(__name__)

# Configuration
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 8085))
API_URL = os.getenv("API_URL", "http://localhost:8083")

# Get current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_file(os.path.join(CURRENT_DIR, 'templates', 'frontend.html'))

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, images)"""
    return send_from_directory(os.path.join(CURRENT_DIR, 'static'), filename)

@app.route('/health')
def frontend_health():
    """Frontend health check"""
    try:
        # Check if API server is reachable
        response = requests.get(f"{API_URL}/health", timeout=5)
        api_healthy = response.status_code == 200
        return jsonify({
            "status": "healthy" if api_healthy else "unhealthy",
            "frontend": "running",
            "api_server": "reachable" if api_healthy else "unreachable",
            "api_url": API_URL
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "frontend": "running", 
            "api_server": "unreachable",
            "api_url": API_URL,
            "error": str(e)
        })

# Proxy API routes to avoid CORS issues
@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api(path):
    """Proxy API calls to the backend server with detailed error logging"""
    try:
        url = f"{API_URL}/{path}"
        logger.info(f"🔄 Proxying {request.method} {url}")
        
        if request.method == 'GET':
            resp = requests.get(url, params=request.args, timeout=180)  # 3 minutes for video generation
        elif request.method == 'POST':
            data = request.get_json()
            logger.info(f"📤 POST data keys: {list(data.keys()) if data else 'None'}")
            resp = requests.post(url, json=data, timeout=180)  # 3 minutes for video generation
        elif request.method == 'PUT':
            resp = requests.put(url, json=request.get_json(), timeout=180)
        elif request.method == 'DELETE':
            resp = requests.delete(url, timeout=180)
        
        logger.info(f"📥 Response: {resp.status_code}")
        
        # Handle streaming responses (MJPEG video streams)
        if resp.headers.get('content-type', '').startswith('multipart/x-mixed-replace'):
            logger.info("📺 Streaming video response - proxying MJPEG stream")
            def generate():
                for chunk in resp.iter_content(chunk_size=1024, decode_unicode=False):
                    if chunk:  # Filter out keep-alive chunks
                        yield chunk
            
            # Create response with proper headers for MJPEG
            response = Response(generate(), content_type=resp.headers['content-type'])
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        
        # Handle JSON responses
        try:
            json_data = resp.json()
            return jsonify(json_data), resp.status_code
        except ValueError:
            # Not JSON, return as text
            return resp.text, resp.status_code
        
    except requests.ConnectionError as e:
        error_msg = f"Cannot connect to API server at {API_URL}"
        logger.error(f"❌ {error_msg}: {e}")
        return jsonify({"error": error_msg}), 503
    except requests.Timeout as e:
        error_msg = "API server timeout"
        logger.error(f"⏱️ {error_msg}: {e}")
        return jsonify({"error": error_msg}), 504
    except requests.RequestException as e:
        error_msg = f"API server error: {str(e)}"
        logger.error(f"🚨 {error_msg}")
        return jsonify({"error": error_msg}), 503
    except Exception as e:
        error_msg = f"Unexpected proxy error: {str(e)}"
        logger.error(f"💥 {error_msg}")
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("🌐 GameCraft Web Frontend Server")
    print("=" * 60)
    print(f"🚀 Starting on port {FRONTEND_PORT}...")
    print(f"🔗 Frontend: http://localhost:{FRONTEND_PORT}")
    print(f"🔗 Backend API: {API_URL}")
    print("📡 Proxying API calls to avoid CORS issues")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=FRONTEND_PORT, debug=False, threaded=True)

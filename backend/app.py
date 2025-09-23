from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import random
import json
import threading
import time
import torch

from utils import load_model, detect_threat
from database import init_db, log_threat, get_stats

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

model = None
model_info = {}
model_lock = threading.Lock()

def load_model_info():
    global model_info
    info_path = os.path.join('models', 'model_info.json')
    try:
        with open(info_path, 'r') as f:
            model_info = json.load(f)
    except Exception as e:
        print(f"Error loading model_info.json: {e}")
        model_info = {
            "model_architecture": "Unknown",
            "parameter_count": 0,
            "number_of_classes": 0,
            "class_labels": [],
            "training_date": None,
            "accuracy_percentage": 0.0,
            "description": ""
        }


def get_or_load_model():
    global model
    if model is None:
        with model_lock:
            if model is None:  # Double-check lock pattern
                try:
                    print("Loading model on demand...")
                    # Use the known working model path
                    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'gcn_threat_detector.pth')
                    print(f"Loading model from: {model_path}")
                    if not os.path.exists(model_path):
                        print(f"Model file not found at {model_path}")
                        return None
                    model = load_model(model_path)
                    if model:
                        print("Model loaded successfully")
                    else:
                        print("Failed to load model")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    return None
    return model


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})


@app.route('/api/stats', methods=['GET'])
def stats():
    stats = get_stats()
    return jsonify(stats)


@app.route('/')
def index():
    return "PRATIRAKSHA Backend is running. Use the frontend dashboard to view data."


@socketio.on('connect')
def handle_connect():
    emit('model_info', model_info)


def listen_to_network_flow():
    time.sleep(1)  # simulate delay
    # Randomly pick a threat type index (0=BENIGN, 1=DOS, 2=RANSOMWARE, 3=BOTNET, 4=MALWARE)
    threat_index = random.randint(0, 4)
    flow = {
        "src_ip": f"192.168.1.{random.randint(2, 254)}",
        "dst_ip": "192.168.1.1",
        "duration": random.uniform(0, 100),
        "protocol": random.choice([6, 17]),  # TCP/UDP
        "src_bytes": random.uniform(0, 10000),
        "dst_bytes": random.uniform(0, 10000),
        "packets": random.randint(1, 100),
        "tcp_flags": random.randint(0, 255),
        "active_time": random.uniform(0, 100),
        "idle_time": random.uniform(0, 100)
    }
    # Exaggerate the first feature to influence the model (if your model is sensitive to this)
    flow["duration"] = float(threat_index) * 20 + random.uniform(0, 10)
    return flow


def monitor_network():
    while True:
        try:
            new_flow = listen_to_network_flow()
            current_model = get_or_load_model()
            if current_model:
                threat = detect_threat(current_model, new_flow)
                if threat:
                    # Log to DB using datetime timestamp
                    db_threat = dict(threat)
                    db_threat['timestamp'] = threat['timestamp']
                    log_threat(db_threat)
                    # Emit to frontend using string timestamp
                    frontend_threat = dict(threat)
                    frontend_threat['timestamp'] = threat['timestamp_str']
                    frontend_threat.pop('timestamp_str', None)
                    socketio.emit('new_threat', frontend_threat)
                    socketio.emit('stats_update', get_stats())
            time.sleep(0.1)  # Add small delay to prevent CPU overload
        except Exception as e:
            print(f"Error in network monitoring: {e}")
            time.sleep(1)  # Wait before retrying


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'build')
    if path != "" and os.path.exists(os.path.join(root_dir, path)):
        return send_from_directory(root_dir, path)
    else:
        return send_from_directory(root_dir, 'index.html')


if __name__ == '__main__':
    try:
        print("Initializing server...")
        init_db()
        load_model_info()
        print("Starting server on port 5002...")
        
        # Start the network monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor_network, daemon=True)
        monitor_thread.start()
        
        # Run the server with socket.io support
        socketio.run(app, host='0.0.0.0', port=5002, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error starting server: {e}", flush=True)
        import traceback
        traceback.print_exc()

import os
import sys
import time
import json
import random
import logging
import threading
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import joblib
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import sqlite3

sys.path.append(".")
from training_gcn_model import NetworkFlowGCN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "pratiraksha_secret_2023"
socketio = SocketIO(app, cors_allowed_origins="*")

threat_detector = None
model_info = None
threat_stats = {
    "total_flows": 0,
    "threats_detected": 0,
    "threats_blocked": 0,
    "benign_flows": 0,
    "uptime_start": datetime.now()
}

class PratirakshaThreatDetector:
    def __init__(self):
        self.model = None
        self.graph_builder = None
        self.model_info = None
        self.is_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        try:
            models_dir = Path("models")
            
            with open(models_dir / "model_info.json", "r") as f:
                self.model_info = json.load(f)
            
            self.graph_builder = joblib.load(models_dir / "graph_builder.pkl")
            
            self.model = NetworkFlowGCN(
                input_dim=self.model_info["input_dim"],
                hidden_dim=self.model_info["hidden_dim"],
                num_classes=self.model_info["num_classes"],
                dropout=0.3
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(models_dir / "gcn_threat_detector.pth", map_location=self.device))
            self.model.eval()
            
            self.is_loaded = True
            logger.info("Production GCN Model Loaded!")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_threat(self, flow_features):
        if not self.is_loaded:
            return {"predicted_class": 0, "confidence": 0.5, "class_name": "Unknown"}
        
        try:
            if isinstance(flow_features, list):
                flow_features = np.array(flow_features).reshape(1, -1)
            elif len(flow_features.shape) == 1:
                flow_features = flow_features.reshape(1, -1)
            
            node_features, edge_index = self.graph_builder.create_single_flow_graph(flow_features)
            prediction = self.model.predict_threat(node_features, edge_index)
            
            class_names = self.model_info["class_names"]
            prediction["class_name"] = class_names[prediction["predicted_class"]]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"predicted_class": 0, "confidence": 0.5, "class_name": "Error"}

def init_database():
    conn = sqlite3.connect("threats.db")
    c = conn.cursor()
    
    c.execute("""CREATE TABLE IF NOT EXISTS threats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  source_ip TEXT,
                  dest_ip TEXT,
                  threat_type TEXT,
                  confidence REAL,
                  status TEXT,
                  details TEXT)""")
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

def generate_realistic_flow():
    flow = []
    
    flow.extend([
        random.randint(1, 65535),
        random.randint(1, 65535),
        random.randint(1, 255),
        random.randint(64, 1500),
        random.randint(1, 1000),
    ])
    
    flow.extend([
        random.randint(1, 10000),
        random.randint(1, 10000),
        random.randint(1, 100),
        random.randint(1, 100),
        random.uniform(0, 1),
    ])
    
    while len(flow) < 50:
        flow.append(random.uniform(0, 1))
    
    return np.array(flow[:50])

def simulate_network_traffic():
    global threat_stats
    
    while True:
        try:
            flow = generate_realistic_flow()
            prediction = threat_detector.predict_threat(flow)
            
            threat_stats["total_flows"] += 1
            
            source_ip = f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
            dest_ip = f"10.0.{random.randint(1,254)}.{random.randint(1,254)}"
            
            if prediction["class_name"] != "Benign":
                threat_stats["threats_detected"] += 1
                
                if prediction["confidence"] > 0.7:
                    threat_stats["threats_blocked"] += 1
                    status = "BLOCKED"
                else:
                    status = "MONITORED"
                
                socketio.emit("new_threat", {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "source_ip": source_ip,
                    "dest_ip": dest_ip,
                    "threat_type": prediction["class_name"],
                    "confidence": f"{prediction['confidence']:.3f}",
                    "status": status
                })
                
                logger.info(f"THREAT DETECTED: {prediction['class_name']} ({prediction['confidence']:.3f}) - {status}")
            else:
                threat_stats["benign_flows"] += 1
            
            uptime = datetime.now() - threat_stats["uptime_start"]
            socketio.emit("stats_update", {
                "total_flows": threat_stats["total_flows"],
                "threats_detected": threat_stats["threats_detected"],
                "threats_blocked": threat_stats["threats_blocked"],
                "benign_flows": threat_stats["benign_flows"],
                "detection_rate": f"{(threat_stats['threats_detected'] / max(threat_stats['total_flows'], 1) * 100):.1f}%",
                "uptime": str(uptime).split(".")[0]
            })
            
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            time.sleep(5)

@app.route("/")
def dashboard():
    return """<!DOCTYPE html>
<html>
<head>
    <title>PRATIRAKSHA-Lite - AI Cybersecurity Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #1e3c72; color: white; }
        .header { background: rgba(0,0,0,0.3); padding: 1rem 2rem; }
        .header h1 { margin: 0; font-size: 2rem; }
        .container { padding: 2rem; display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; max-width: 1200px; margin: 0 auto; }
        .card { background: rgba(255,255,255,0.1); border-radius: 12px; padding: 1.5rem; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
        .stat-item { background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; text-align: center; }
        .stat-number { font-size: 2rem; font-weight: bold; color: #4fc3f7; }
        .stat-label { font-size: 0.9rem; opacity: 0.8; }
        .threat-log { max-height: 400px; overflow-y: auto; background: rgba(0,0,0,0.2); border-radius: 8px; padding: 1rem; }
        .threat-item { padding: 0.5rem; margin-bottom: 0.5rem; border-left: 4px solid #f44336; background: rgba(244,67,54,0.1); border-radius: 4px; font-size: 0.9rem; }
        .model-info { background: rgba(76,175,80,0.2); border: 1px solid #4caf50; }
        .footer { text-align: center; padding: 2rem; opacity: 0.7; }
    </style>
</head>
<body>
    <div class="header">
        <h1>PRATIRAKSHA-Lite AI Cybersecurity Dashboard</h1>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>Live Statistics</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number" id="total-flows">0</div>
                    <div class="stat-label">Total Flows</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="threats-detected">0</div>
                    <div class="stat-label">Threats Detected</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="threats-blocked">0</div>
                    <div class="stat-label">Threats Blocked</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="detection-rate">0%</div>
                    <div class="stat-label">Detection Rate</div>
                </div>
            </div>
        </div>
        
        <div class="card model-info">
            <h2>AI Model Status</h2>
            <p><strong>Model Type:</strong> Graph Convolutional Network (GCN)</p>
            <p><strong>Status:</strong> <span style="color: #4caf50;">Active & Monitoring</span></p>
            <p><strong>Accuracy:</strong> 79.2%</p>
        </div>
        
        <div class="card" style="grid-column: span 2;">
            <h2>Real-Time Threat Detection Log</h2>
            <div class="threat-log" id="threat-log">
                <div style="text-align: center; opacity: 0.7; padding: 2rem;">
                    Monitoring network traffic... Threats will appear here.
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>PRATIRAKSHA-Lite 2023 - AI-Powered Network Threat Detection System</p>
    </div>
    
    <script>
        const socket = io();
        
        socket.on("stats_update", function(data) {
            document.getElementById("total-flows").textContent = data.total_flows;
            document.getElementById("threats-detected").textContent = data.threats_detected;
            document.getElementById("threats-blocked").textContent = data.threats_blocked;
            document.getElementById("detection-rate").textContent = data.detection_rate;
        });
        
        socket.on("new_threat", function(data) {
            const threatLog = document.getElementById("threat-log");
            
            if (threatLog.children.length === 1 && threatLog.children[0].textContent.includes("Monitoring")) {
                threatLog.innerHTML = "";
            }
            
            const threatItem = document.createElement("div");
            threatItem.className = "threat-item";
            threatItem.innerHTML = `
                <strong>${data.timestamp}</strong> - 
                <strong style="color: #ff5722;">${data.threat_type}</strong> detected<br>
                <small>${data.source_ip} --&gt; ${data.dest_ip} | Confidence: ${data.confidence} | Status: <strong>${data.status}</strong></small>
            `;
            
            threatLog.insertBefore(threatItem, threatLog.firstChild);
            
            while (threatLog.children.length > 50) {
                threatLog.removeChild(threatLog.lastChild);
            }
        });
    </script>
</body>
</html>"""

@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")

@socketio.on("disconnect") 
def handle_disconnect():
    logger.info("Client disconnected")

@app.route("/api/stats")
def api_stats():
    uptime = datetime.now() - threat_stats["uptime_start"]
    return jsonify({
        **threat_stats,
        "uptime": str(uptime).split(".")[0],
        "detection_rate": threat_stats["threats_detected"] / max(threat_stats["total_flows"], 1) * 100,
        "model_loaded": threat_detector.is_loaded if threat_detector else False
    })

if __name__ == "__main__":
    logger.info("PRATIRAKSHA-Lite initializing...")
    
    init_database()
    
    threat_detector = PratirakshaThreatDetector()
    model_loaded = threat_detector.load_model()
    
    if model_loaded:
        model_info = threat_detector.model_info
        logger.info("PRATIRAKSHA-Lite initialized successfully")
        
        simulation_thread = threading.Thread(target=simulate_network_traffic, daemon=True)
        simulation_thread.start()
        logger.info("Background threat simulation started")
        
        logger.info("Access dashboard at: http://localhost:5000")
        socketio.run(app, host="0.0.0.0", port=5000, debug=False)
    else:
        logger.error("Failed to initialize PRATIRAKSHA-Lite")
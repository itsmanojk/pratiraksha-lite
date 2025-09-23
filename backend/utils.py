import os
import torch
import datetime

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os
import torch
import datetime
import torch.nn.functional as F

from models.gcn_threat_detector import NetworkFlowGCN

def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Create a new instance of the model with matching architecture
        model = NetworkFlowGCN(
            input_dim=50,  # Input dimension from training
            hidden_dim=128,  # Hidden dimension from saved model
            num_classes=5,  # Number of threat classes
            dropout=0.3
        )
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        if model is None:
            raise ValueError("Model loaded as None")
            
        model.eval()  # Set to evaluation mode
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def detect_threat(model, network_flow):
    try:
        if model is None:
            print("Model is not loaded, cannot detect threats")
            return None

        # Convert network flow to features
        features = torch.zeros(1, 50)  # Initialize feature tensor
        
        # Extract numeric features from network flow
        numeric_features = [
            float(network_flow.get("duration", 0)),
            float(network_flow.get("protocol", 0)),
            float(network_flow.get("src_bytes", 0)),
            float(network_flow.get("dst_bytes", 0)),
            float(network_flow.get("packets", 0)),
            float(network_flow.get("tcp_flags", 0)),
            float(network_flow.get("active_time", 0)),
            float(network_flow.get("idle_time", 0))
        ]
        
        # Fill in the first few positions with actual features
        for i, value in enumerate(numeric_features):
            if i < 50:  # Ensure we don't exceed tensor size
                features[0, i] = value
                
        # Normalize features to prevent extreme values
        features = torch.clamp(features, -10, 10)
        
        # Create a simple edge index for single node
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            try:
                output = model(features, edge_index)
                if output is None:
                    raise ValueError("Model output is None")
                    
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                threat_types = ['BENIGN', 'DOS', 'RANSOMWARE', 'BOTNET', 'MALWARE']
                
                now = datetime.datetime.utcnow()
                return {
                    "timestamp": now,  # for DB
                    "timestamp_str": now.isoformat(),  # for frontend
                    "source_ip": network_flow["src_ip"],
                    "dest_ip": network_flow["dst_ip"],
                    "threat_type": threat_types[predicted_class],
                    "confidence": float(confidence),
                    "status": "DETECTED" if predicted_class > 0 else "BENIGN"
                }
            except Exception as e:
                print(f"Error in model inference: {str(e)}")
                return None
    except Exception as e:
        print(f"Error in threat detection: {str(e)}")
        return None

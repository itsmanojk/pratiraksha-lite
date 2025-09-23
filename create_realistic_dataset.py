import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

def create_realistic_cicids_dataset():
    """Create a realistic CIC-IDS2017 style dataset for immediate GCN training"""
    print("🚀 Creating realistic CIC-IDS2017 style dataset...")
    print("📊 This mimics the actual CIC-IDS2017 structure and features")

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Dataset parameters - larger for better training
    n_samples = 100000  # 100K samples for robust training

    print(f"📈 Generating {n_samples:,} network flow samples...")

    # CIC-IDS2017 actual feature names (79 features total)
    features = {}

    # Basic flow features
    features['Destination Port'] = np.random.choice([80, 443, 22, 21, 25, 53, 8080, 3389, 135, 445], n_samples)
    features['Flow Duration'] = np.abs(np.random.exponential(2000000, n_samples))  # microseconds
    features['Total Fwd Packets'] = np.random.poisson(15, n_samples)
    features['Total Backward Packets'] = np.random.poisson(10, n_samples)
    features['Total Length of Fwd Packets'] = np.random.gamma(2, 500, n_samples)
    features['Total Length of Bwd Packets'] = np.random.gamma(2, 300, n_samples)

    # Packet size statistics
    features['Fwd Packet Length Max'] = np.random.gamma(3, 400, n_samples)
    features['Fwd Packet Length Min'] = np.random.exponential(50, n_samples)
    features['Fwd Packet Length Mean'] = np.random.gamma(2, 200, n_samples)
    features['Fwd Packet Length Std'] = np.random.gamma(1, 100, n_samples)

    features['Bwd Packet Length Max'] = np.random.gamma(3, 350, n_samples)
    features['Bwd Packet Length Min'] = np.random.exponential(40, n_samples)
    features['Bwd Packet Length Mean'] = np.random.gamma(2, 180, n_samples)
    features['Bwd Packet Length Std'] = np.random.gamma(1, 90, n_samples)

    # Flow bytes/sec and packet/s
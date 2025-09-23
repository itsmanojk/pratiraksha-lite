import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_sample_dataset():
    """Create a sample network intrusion dataset for immediate training"""
    print("🔄 Creating sample network intrusion dataset...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Dataset parameters
    n_samples = 50000  # Larger sample for better training
    n_features = 20

    print(f"📊 Generating {n_samples:,} network flow samples...")

    # Create realistic network flow features
    data = {}

    # Basic flow features
    data['duration'] = np.random.exponential(2.0, n_samples)  # Flow duration in seconds
    data['total_fwd_packets'] = np.random.poisson(15, n_samples)  # Forward packets
    data['total_bwd_packets'] = np.random.poisson(10, n_samples)  # Backward packets
    data['total_length_fwd_packets'] = np.random.gamma(2, 500, n_samples)  # Forward bytes
    data['total_length_bwd_packets'] = np.random.gamma(2, 300, n_samples)  # Backward bytes

    # Packet size statistics
    data['fwd_packet_length_max'] = np.random.gamma(3, 200, n_samples)
    data['fwd_packet_length_min'] = np.random.exponential(50, n_samples)
    data['fwd_packet_length_mean'] = np.random.gamma(2, 150, n_samples)
    data['fwd_packet_length_std'] = np.random.gamma(1, 80, n_samples)

    data['bwd_packet_length_max'] = np.random.gamma(3, 180, n_samples)
    data['bwd_packet_length_min'] = np.random.exponential(40, n_samples)
    data['bwd_packet_length_mean'] = np.random.gamma(2, 120, n_samples)
    data['bwd_packet_length_std'] = np.random.gamma(1, 70, n_samples)

    # Flow statistics
    data['flow_bytes_per_sec'] = np.random.gamma(2, 1000, n_samples)
    data['flow_packets_per_sec'] = np.random.gamma(2, 10, n_samples)
    data['flow_iat_mean'] = np.random.exponential(1000, n_samples)  # Inter-arrival time
    data['flow_iat_std'] = np.random.gamma(2, 50)
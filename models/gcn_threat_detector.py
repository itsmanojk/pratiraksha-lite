import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkFlowGCN(nn.Module):
    """Graph Convolutional Network for Network Threat Detection"""

    def __init__(self, input_dim, hidden_dim=128, num_classes=5, dropout=0.3):
        super(NetworkFlowGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # GCN layers with adjusted dimensions to match saved model
        self.conv1 = GCNConv(input_dim, hidden_dim)  # 128 output
        self.conv2 = GCNConv(hidden_dim, 64)  # 64 output
        self.conv3 = GCNConv(64, 32)  # 32 output

        # Attention layer with matching dimensions
        self.attention = GATConv(32, 32, heads=4, concat=False)  # 32 output

        # Classification layers with matching dimensions
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)  # 5 output classes
        )

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 128
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)

        logger.info(f"🧠 GCN Model initialized:")
        logger.info(f"   Input dim: {input_dim}")
        logger.info(f"   Hidden dim: {hidden_dim}")
        logger.info(f"   Output classes: {num_classes}")

    def forward(self, x, edge_index, batch=None):
        """Forward pass through the network"""
        # GCN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        
        # Attention layer
        x = F.relu(self.attention(x, edge_index))
        
        # If we have batch information, use it for pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # If no batch info, we're dealing with a single graph
            x = torch.mean(x, dim=0).unsqueeze(0)
        
        # Classification layers
        x = self.classifier(x)
        
        return x
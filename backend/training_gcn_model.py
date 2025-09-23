import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkFlowGCN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=6, dropout=0.3):
        super(NetworkFlowGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        
        self.attention = GATConv(hidden_dim // 4, hidden_dim // 4, heads=4, concat=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 8, num_classes)
        )
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        
        logger.info(f"GCN Model initialized:")
        logger.info(f"   Input dim: {input_dim}")
        logger.info(f"   Hidden dim: {hidden_dim}")
        logger.info(f"   Output classes: {num_classes}")
        
    def forward(self, x, edge_index, batch=None):
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.attention(x, edge_index)
        x = F.relu(x)
        
        # FIXED: For node-level predictions, don't do global pooling in training
        if batch is not None:
            x = global_mean_pool(x, batch)
            return F.log_softmax(x, dim=1)
        else:
            # For node classification (training), return node-level predictions
            return F.log_softmax(x, dim=1)
    
    def predict_threat(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            output = self.forward(x, edge_index)
            
            # For single sample prediction, aggregate the nodes
            if len(x) > 1:
                # Multiple nodes - take mean
                output = torch.mean(output, dim=0, keepdim=True)
            
            probabilities = torch.exp(output).squeeze().numpy()
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[predicted_class]
            
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': probabilities.tolist()
        }

class NetworkGraphBuilder:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        
    def create_graph_from_flows(self, flows_df, target_col='Label'):
        logger.info(f"Creating graphs from {len(flows_df)} flows...")
        
        features = flows_df.drop(columns=[target_col])
        labels = flows_df[target_col]
        
        self.feature_names = features.columns.tolist()
        self.class_names = labels.unique().tolist()
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        scaled_features = self.scaler.fit_transform(features)
        
        node_features = torch.FloatTensor(scaled_features)
        node_labels = torch.LongTensor(encoded_labels)
        
        edge_index = self._create_edges_knn(scaled_features, k=10)
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=node_labels
        )
        
        logger.info(f"Graph created:")
        logger.info(f"   Nodes: {data.num_nodes}")
        logger.info(f"   Edges: {data.num_edges}")
        logger.info(f"   Features: {data.num_node_features}")
        logger.info(f"   Classes: {len(self.class_names)}")
        
        return data, self.class_names
        
    def _create_edges_knn(self, features, k=10):
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(features)
        distances, indices = nbrs.kneighbors(features)
        
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:
                edges.append([i, neighbor])
                edges.append([neighbor, i])
        
        edge_index = torch.LongTensor(edges).t().contiguous()
        
        return edge_index
    
    def create_single_flow_graph(self, flow_features):
        if self.scaler is None:
            raise ValueError("GraphBuilder not fitted. Train the model first.")
            
        if len(flow_features.shape) == 1:
            flow_features = flow_features.reshape(1, -1)
            
        scaled_features = self.scaler.transform(flow_features)
        node_features = torch.FloatTensor(scaled_features)
        
        edge_index = torch.LongTensor([[0], [0]])
        
        return node_features, edge_index

class ThreatDetectionTrainer:
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        pred = out[data.train_mask].argmax(dim=1)
        train_acc = (pred == data.y[data.train_mask]).float().mean()
        
        return loss.item(), train_acc.item()
    
    def validate(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            
            val_loss = self.criterion(out[data.val_mask], data.y[data.val_mask])
            
            pred = out[data.val_mask].argmax(dim=1)
            val_acc = (pred == data.y[data.val_mask]).float().mean()
            
        return val_loss.item(), val_acc.item()
    
    def train(self, data, epochs=100, patience=10):
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(data)
            
            val_loss, val_acc = self.validate(data)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_gcn_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                           f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load('best_gcn_model.pth'))
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return self.history
    
    def test(self, data):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out[data.test_mask].argmax(dim=1)
            test_acc = (pred == data.y[data.test_mask]).float().mean()
            
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        return test_acc.item()

def create_train_val_test_masks(num_nodes, train_ratio=0.6, val_ratio=0.2):
    indices = torch.randperm(num_nodes)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask

if __name__ == "__main__":
    logger.info("Testing GCN model with sample data...")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    features = np.random.randn(n_samples, n_features)
    labels = np.random.choice(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'], n_samples)
    
    df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
    df['Label'] = labels
    
    graph_builder = NetworkGraphBuilder()
    data, class_names = graph_builder.create_graph_from_flows(df)
    
    data.train_mask, data.val_mask, data.test_mask = create_train_val_test_masks(data.num_nodes)
    
    model = NetworkFlowGCN(
        input_dim=data.num_node_features,
        hidden_dim=64,
        num_classes=len(class_names)
    )
    
    trainer = ThreatDetectionTrainer(model)
    history = trainer.train(data, epochs=50)
    
    test_acc = trainer.test(data)
    
    logger.info("GCN model test completed successfully!")
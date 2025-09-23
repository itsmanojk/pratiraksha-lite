import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append('.')
from training_gcn_model import NetworkFlowGCN, NetworkGraphBuilder, ThreatDetectionTrainer, create_train_val_test_masks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ThreatDetectionPipeline:

    def __init__(self, config=None):
        self.config = config or {
            'hidden_dim': 64,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 5e-4,
            'epochs': 100,
            'patience': 15,
            'batch_size': 32,
            'test_size': 0.3,
            'val_size': 0.2
        }

        self.model = None
        self.graph_builder = None
        self.trainer = None
        self.class_names = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info("PRATIRAKSHA-Lite GCN Training Pipeline Initialized")
        logger.info(f"Device: {self.device}")

    def load_dataset(self, dataset_path):
        logger.info(f"Loading dataset from: {dataset_path}")

        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            else:
                raise ValueError("Only CSV files are supported")

            logger.info("Dataset loaded successfully")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

            possible_labels = ['Label', 'label', 'Attack', 'attack', 'class', 'Class', 'target', 'Target']
            label_col = None

            for col in possible_labels:
                if col in df.columns:
                    label_col = col
                    break

            if label_col is None:
                for col in df.columns:
                    if df[col].dtype == 'object' or df[col].nunique() < 20:
                        unique_vals = df[col].unique()
                        attack_keywords = ['dos', 'ddos', 'malware', 'benign', 'normal', 'ransomware', 'bot']
                        if any(any(keyword.lower() in str(val).lower() for keyword in attack_keywords) for val in unique_vals):
                            label_col = col
                            break

            if label_col is None:
                label_col = df.columns[-1]
                logger.warning(f"No obvious label column found. Using last column: {label_col}")

            logger.info(f"Label column: {label_col}")

            class_counts = df[label_col].value_counts()
            for class_name, count in class_counts.items():
                logger.info(f"   {class_name}: {count} ({count/len(df)*100:.1f}%)")

            return df, label_col

        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Dataset file is empty: {dataset_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def preprocess_data(self, df, label_col):
        logger.info("Preprocessing data...")

        missing_count = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_count}")
        if missing_count > 0:
            for col in df.columns:
                if col != label_col:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
            logger.info("Missing values handled")

        initial_size = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_size - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != label_col:
                pos_inf_mask = np.isposinf(df[col])
                neg_inf_mask = np.isneginf(df[col])

                if pos_inf_mask.any():
                    finite_max = df.loc[~pos_inf_mask, col].max()
                    df.loc[pos_inf_mask, col] = finite_max if not pd.isna(finite_max) else 0

                if neg_inf_mask.any():
                    finite_min = df.loc[~neg_inf_mask, col].min()
                    df.loc[neg_inf_mask, col] = finite_min if not pd.isna(finite_min) else 0

        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != label_col]

        if len(categorical_cols) > 0:
            logger.info(f"Converting categorical columns: {categorical_cols}")
            for col in categorical_cols:
                unique_vals = df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df[col] = df[col].map(mapping)

        class_counts = df[label_col].value_counts()
        min_samples = class_counts.min()
        max_samples = class_counts.max()
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')

        if imbalance_ratio > 20:
            logger.info(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            logger.info("Applying balanced sampling...")

            balanced_dfs = []
            target_samples = min(min_samples * 3, max_samples // 2)

            for class_name in class_counts.index:
                class_df = df[df[label_col] == class_name]
                current_count = len(class_df)

                if current_count < target_samples:
                    n_samples = min(target_samples, current_count * 5)
                    sampled_df = class_df.sample(n=n_samples, replace=True, random_state=42)
                else:
                    n_samples = min(current_count, target_samples * 2)
                    sampled_df = class_df.sample(n=n_samples, random_state=42)

                balanced_dfs.append(sampled_df)

            df = pd.concat(balanced_dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info(f"Dataset balanced. New shape: {df.shape}")

        feature_cols = [col for col in df.columns if col != label_col]
        for col in feature_cols:
            if df[col].dtype == 'object':
                logger.warning(f"Converting remaining object column to numeric: {col}")
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        for col in feature_cols:
            if df[col].min() < 0:
                df[col] = df[col] - df[col].min()

        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        logger.info(f"Features: {len(feature_cols)}, Classes: {df[label_col].nunique()}")

        return df

    def train_model(self, dataset_path):
        logger.info("Starting GCN model training...")

        df, label_col = self.load_dataset(dataset_path)
        df = self.preprocess_data(df, label_col)

        self.graph_builder = NetworkGraphBuilder()
        data, self.class_names = self.graph_builder.create_graph_from_flows(df, label_col)

        data.train_mask, data.val_mask, data.test_mask = create_train_val_test_masks(
            data.num_nodes,
            train_ratio=1 - self.config['test_size'] - self.config['val_size'],
            val_ratio=self.config['val_size']
        )

        data = data.to(self.device)

        self.model = NetworkFlowGCN(
            input_dim=data.num_node_features,
            hidden_dim=self.config['hidden_dim'],
            num_classes=len(self.class_names),
            dropout=self.config['dropout']
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")

        self.trainer = ThreatDetectionTrainer(self.model, self.device)

        self.trainer.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        logger.info(f"Starting training for {self.config['epochs']} epochs...")
        history = self.trainer.train(
            data,
            epochs=self.config['epochs'],
            patience=self.config['patience']
        )

        test_accuracy = self.trainer.test(data)

        self.evaluate_model(data)
        self.save_model()
        self.plot_training_history(history)

        logger.info("Training completed successfully!")

        return {
            'test_accuracy': test_accuracy,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_parameters': total_params,
            'training_history': history
        }

    def evaluate_model(self, data):
        logger.info("Evaluating model performance...")

        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)

            test_pred = out[data.test_mask].argmax(dim=1).cpu().numpy()
            test_true = data.y[data.test_mask].cpu().numpy()

            try:
                report = classification_report(
                    test_true, test_pred,
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0
                )

                logger.info("Classification Report:")
                for class_name in self.class_names:
                    if class_name in report:
                        metrics = report[class_name]
                        logger.info(f"   {class_name:20s}: Precision: {metrics['precision']:.3f}, "
                                   f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")

                logger.info(f"Overall Accuracy: {report['accuracy']:.3f}")
                if 'macro avg' in report:
                    logger.info(f"Macro F1-Score: {report['macro avg']['f1-score']:.3f}")

            except Exception as e:
                logger.warning(f"Could not generate classification report: {e}")
                logger.info(f"Test Accuracy: {accuracy_score(test_true, test_pred):.3f}")

            try:
                cm = confusion_matrix(test_true, test_pred)

                plt.figure(figsize=(max(8, len(self.class_names)), max(6, len(self.class_names))))
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names
                )
                plt.title('Confusion Matrix - GCN Threat Detection')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()

                os.makedirs('logs', exist_ok=True)
                plt.savefig('logs/confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Confusion matrix saved: logs/confusion_matrix.png")

                if 'report' in locals():
                    with open('logs/evaluation_results.json', 'w') as f:
                        json.dump(report, f, indent=2)
                    logger.info("Evaluation results saved: logs/evaluation_results.json")

            except Exception as e:
                logger.warning(f"Could not generate confusion matrix: {e}")

    def plot_training_history(self, history):
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            epochs = range(len(history['train_loss']))

            ax1.plot(epochs, history['train_loss'], label='Training Loss', color='blue', linewidth=2)
            ax1.plot(epochs, history['val_loss'], label='Validation Loss', color='red', linewidth=2)
            ax1.set_title('Model Loss During Training')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(epochs, history['train_acc'], label='Training Accuracy', color='blue', linewidth=2)
            ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', color='red', linewidth=2)
            ax2.set_title('Model Accuracy During Training')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            os.makedirs('logs', exist_ok=True)
            plt.savefig('logs/training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Training history plot saved: logs/training_history.png")

        except Exception as e:
            logger.warning(f"Could not plot training history: {e}")

    def save_model(self):
        logger.info("Saving trained model...")

        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            torch.save(self.model.state_dict(), models_dir / "gcn_threat_detector.pth")

            joblib.dump(self.graph_builder, models_dir / "graph_builder.pkl")

            model_info = {
                'model_type': 'NetworkFlowGCN',
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_classes': self.model.num_classes,
                'class_names': self.class_names,
                'config': self.config,
                'training_date': datetime.now().isoformat(),
                'device': str(self.device),
                'pytorch_version': torch.__version__
            }

            with open(models_dir / "model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)

            logger.info("Model saved successfully!")
            logger.info(f"Model files saved in: {models_dir}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise


def main():
    logger.info("PRATIRAKSHA-Lite GCN Training Started")
    logger.info(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = {
        'hidden_dim': 128,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
        'epochs': 200,
        'patience': 25,
        'test_size': 0.2,
        'val_size': 0.2
    }

    pipeline = ThreatDetectionPipeline(config)

    dataset_paths = [
        "data/PRATIRAKSHA_ransomware_dataset.csv",
        "data/CICIDS2017_sample.csv",
        "data/network_intrusion_dataset.csv",
        "data/sample_network_dataset.csv",
        "PRATIRAKSHA_ransomware_dataset.csv",
        "CICIDS2017_sample.csv",
        "network_intrusion_dataset.csv",
        "sample_network_dataset.csv"
    ]

    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            logger.info(f"Found dataset: {path}")
            break

    if dataset_path is None:
        logger.info("No dataset found. Creating sample dataset...")
        try:
            if os.path.exists('create_ransomware_dataset.py'):
                logger.info("Creating ransomware-focused dataset...")
                os.system('python create_ransomware_dataset.py')
                dataset_path = "data/PRATIRAKSHA_ransomware_dataset.csv"
            elif os.path.exists('create_realistic_dataset.py'):
                logger.info("Creating realistic dataset...")
                os.system('python create_realistic_dataset.py')
                dataset_path = "data/CICIDS2017_sample.csv"
            elif os.path.exists('create_sample_dataset.py'):
                logger.info("Creating basic sample dataset...")
                os.system('python create_sample_dataset.py')
                dataset_path = "data/network_intrusion_dataset.csv"
            else:
                raise FileNotFoundError("No dataset creator found")

        except Exception as e:
            logger.error(f"Could not create dataset: {e}")
            logger.error("Please ensure you have the dataset creation scripts available")
            return

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        logger.error("Please create a dataset first using one of the dataset creation scripts")
        return

    try:
        logger.info(f"Using dataset: {dataset_path}")
        results = pipeline.train_model(dataset_path)

        logger.info("")
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Classes Detected: {results['num_classes']}")
        logger.info(f"Model Parameters: {results['model_parameters']:,}")
        logger.info(f"Training Duration: {len(results['training_history']['train_loss'])} epochs")
        logger.info("")
        logger.info("Generated Files:")
        logger.info("   - models/gcn_threat_detector.pth")
        logger.info("   - models/graph_builder.pkl")
        logger.info("   - models/model_info.json")
        logger.info("   - logs/confusion_matrix.png")
        logger.info("   - logs/training_history.png")
        logger.info("   - logs/evaluation_results.json")
        logger.info("   - training.log")

        logger.info("")
        logger.info("Testing model inference...")

        n_features = pipeline.graph_builder.scaler.n_features_in_
        sample_flow = np.random.randn(1, n_features)
        node_features, edge_index = pipeline.graph_builder.create_single_flow_graph(sample_flow)

        prediction = pipeline.model.predict_threat(node_features, edge_index)
        predicted_class_name = results['class_names'][prediction['predicted_class']]

        logger.info(f"Sample prediction: {predicted_class_name} (confidence: {prediction['confidence']:.3f})")

        logger.info("")
        logger.info("PRATIRAKSHA-Lite model is ready for deployment!")
        logger.info("Next step: python app.py")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Please check the error messages above and ensure all dependencies are installed")
        raise


if __name__ == "__main__":
    main()
    
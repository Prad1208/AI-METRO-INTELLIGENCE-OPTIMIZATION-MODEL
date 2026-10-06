"""
Deep Autoencoder for Rolling Stock & Track Predictive Maintenance.
Unsupervised reconstruction-based anomaly detection on high-frequency vibration,
motor current, and bearing temperature telemetry.
"""
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from config import ANOMALY_CONFIG, CHECKPOINT_DIR, DATA_DIR


class DeepAutoencoder(nn.Module):
    """
    Symmetric Deep Autoencoder for sensor telemetry reconstruction.
    """

    def __init__(self, input_dim: int = 5, bottleneck_dim: int = 4):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, bottleneck_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class PredictiveMaintenanceEngine:
    """
    Trains and infers mechanical fault anomalies across the metro fleet.
    """

    def __init__(self, input_dim: int = 5):
        self.input_dim = input_dim
        self.model = DeepAutoencoder(input_dim=input_dim)
        self.scaler = StandardScaler()
        self.threshold = 0.0

    def fit(self, df: pd.DataFrame, epochs: int = 20, batch_size: int = 64) -> Dict[str, Any]:
        """
        Fits autoencoder on predominantly normal operational telemetry (unsupervised).
        """
        feature_cols = ["vibration_x", "vibration_y", "vibration_z", "motor_current_a", "bearing_temp_c"]
        X = df[feature_cols].values
        y = df["is_anomaly"].values

        # Split train (75%) and test (25%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train only on normal instances in training split (pure unsupervised anomaly detection)
        normal_mask = (y_train == 0)
        X_train_normal = X_train_scaled[normal_mask]

        train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(train_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=ANOMALY_CONFIG["learning_rate"], weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.model.train()
        print(f"🔧 Training Predictive Maintenance Autoencoder ({epochs} epochs)...")
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (batch_x,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0 or epoch == 1:
                avg_l = total_loss / len(loader)
                print(f"Epoch {epoch:02d}/{epochs:02d} | Reconstruction MSE: {avg_l:.5f}")

        # Set Anomaly Threshold at 95th percentile of normal training errors
        self.model.eval()
        with torch.no_grad():
            train_recon = self.model(train_tensor).numpy()
            train_errors = np.mean((X_train_normal - train_recon) ** 2, axis=1)
            self.threshold = float(np.percentile(train_errors, 96.0))

        # Evaluate on Test Set
        with torch.no_grad():
            test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            test_recon = self.model(test_tensor).numpy()
            test_errors = np.mean((X_test_scaled - test_recon) ** 2, axis=1)
            y_pred = (test_errors > self.threshold).astype(int)

        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, test_errors))

        print(f"🎯 Maintenance Anomaly Model Evaluated -> ROC-AUC: {auc:.4f} | F1-Score: {f1:.4f} | Threshold: {self.threshold:.4f}")

        # Save model
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_mean": torch.tensor(self.scaler.mean_, dtype=torch.float32),
            "scaler_scale": torch.tensor(self.scaler.scale_, dtype=torch.float32),
            "threshold": self.threshold
        }, CHECKPOINT_DIR / "autoencoder_pdm.pt")

        return {
            "roc_auc": round(auc, 4),
            "f1_score": round(f1, 4),
            "threshold": round(self.threshold, 4),
            "test_size": len(y_test)
        }

    def predict(self, sample_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infers anomaly labels (0: Normal, 1: Mechanical Defect) and raw reconstruction error scores.
        """
        self.model.eval()
        scaled = self.scaler.transform(sample_features)
        with torch.no_grad():
            t = torch.tensor(scaled, dtype=torch.float32)
            recon = self.model(t).numpy()
            errors = np.mean((scaled - recon) ** 2, axis=1)
            labels = (errors > self.threshold).astype(int)
        return labels, errors


if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "telemetry_data.csv")
    engine = PredictiveMaintenanceEngine()
    metrics = engine.fit(df)
    print("Training Complete:", metrics)

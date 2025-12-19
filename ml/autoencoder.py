import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetector:
    """
    Unsupervised anomaly detection that
    learns normal motion patterns, flags deviations
    """
    def __init__(self, input_dim=32, latent_dim=8):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.scaler = StandardScaler()
        self.model = None
        self.encoder = None
        self.threshold = None
        self.device = torch.device("cpu")

    def build_model(self):
        """A tiny autoencoder for edge deployment"""
        self.model = Autoencoder(self.input_dim, self.latent_dim).to(self.device)
        self.encoder = self.model.encoder

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, normal_data, epochs=50):
        """Training on normal operation data"""
        # Scale
        X_scaled = self.scaler.fit_transform(normal_data)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, X_tensor)
            loss.backward()
            self.optimizer.step()

        # Set anomaly threshold (95th percentile of reconstruction error)
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor).cpu().numpy()

        mse = np.mean((X_scaled - reconstructions) ** 2, axis=1)
        self.threshold = np.percentile(mse, 95)

        return None

    def detect(self, features):
        """Detect anomaly in new sample"""
        X_scaled = self.scaler.transform(features.reshape(1, -1))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(X_tensor).cpu().numpy()

        mse = np.mean((X_scaled - reconstruction) ** 2)

        anomaly_score = min(mse / self.threshold, 1.0) if self.threshold > 0 else 0

        return {
            'is_anomaly': mse > self.threshold,
            'anomaly_score': float(anomaly_score),
            'reconstruction_error': float(mse),
            'threshold': float(self.threshold)
        }

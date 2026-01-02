import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnxruntime as ort


class Autoencoder(nn.Module):
    """
    Physics-aware autoencoder for anomaly detection
    Learns normal motion patterns from physics-based features
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder: compress physics features to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: reconstruct physics features from latent space
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
    Physics-based anomaly detection that
    identifies unusual motion patterns using physical feature analysis
    """
    def __init__(self, input_dim=32, latent_dim=8):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.scaler = StandardScaler()
        self.model = None
        self.encoder = None
        self.device = torch.device("cpu")

    def build_model(self):
        """Build physics-aware autoencoder"""
        self.model = Autoencoder(self.input_dim, self.latent_dim).to(self.device)
        self.encoder = self.model.encoder
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, normal_data, epochs=50):
        """Train on physics-based motion features"""
        # Scale physics features
        X_scaled = self.scaler.fit_transform(normal_data)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, X_tensor)
            loss.backward()
            self.optimizer.step()

        # Set anomaly threshold based on physics-based reconstruction
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            mse = np.mean((X_scaled - reconstructions.cpu().numpy()) ** 2, axis=1)
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
    
    def export_onnx(self, model_path="models/anomaly_detector.onnx"):
        """Export trained model to ONNX for edge deployment"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create dummy input for tracing
        dummy_input = torch.randn(1, self.input_dim).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"✅ Model exported to {model_path}")
        return model_path
    
    def load_onnx(self, model_path="models/anomaly_detector.onnx"):
        """Load ONNX model for inference"""
        try:
            # Load ONNX model
            self.onnx_session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Get input/output info
            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_name = self.onnx_session.get_outputs()[0].name
            
            print(f"✅ ONNX model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            return False
    
    def detect_onnx(self, features):
        """Detect anomaly using ONNX model (faster for edge)"""
        if not hasattr(self, 'onnx_session'):
            # Fallback to PyTorch if ONNX not loaded
            return self.detect(features)
        
        # Prepare input
        X_scaled = self.scaler.transform(features.reshape(1, -1))
        input_data = X_scaled.astype(np.float32)
        
        # Run inference
        outputs = self.onnx_session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        
        reconstruction = outputs[0]
        mse = np.mean((X_scaled - reconstruction) ** 2)
        
        anomaly_score = min(mse / self.threshold, 1.0) if self.threshold > 0 else 0
        
        return {
            'is_anomaly': mse > self.threshold,
            'anomaly_score': float(anomaly_score),
            'reconstruction_error': float(mse),
            'threshold': float(self.threshold),
            'inference_engine': 'onnx'
        }

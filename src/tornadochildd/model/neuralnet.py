"""Neural network for binary classification on the diabetes dataset."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


class TorchNet(nn.Module):
    """Feedforward neural network with two hidden layers."""

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, output_dim),
        )

        # Include the training means and scales in the exported model.
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))

    def fit_scaler(self, features):
        """Learn the mean and scale of each feature from the training data."""
        scaler = StandardScaler().fit(features)

        self.feature_mean.copy_(torch.as_tensor(scaler.mean_))
        self.feature_scale.copy_(torch.as_tensor(scaler.scale_))

    def forward(self, features):
        """Standardize the raw features and calculate the output logits."""
        standardized = (features - self.feature_mean) / self.feature_scale
        return self.network(standardized)

    def fit_model(self, features, labels, epochs=1000, learning_rate=0.01):
        """Train with Adam and return the binary cross-entropy loss per epoch."""
        x = torch.as_tensor(features, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.float32)

        # For one output, give the labels the same shape as the logits.
        if y.ndim == 1:
            y = y.unsqueeze(1)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        loss_history = []

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        return loss_history

    def plot_loss(self, loss_history):
        """Plot training loss against epoch number."""
        epochs = range(1, len(loss_history) + 1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, loss_history, color="steelblue", linewidth=1.8)
        ax.set_title("TorchNet training loss on the diabetes dataset")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary cross-entropy loss")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()

    def save_onnx(self, filename):
        """Save the trained network and its scaling parameters in ONNX format."""
        self.eval()
        example_input = torch.zeros_like(self.feature_mean).unsqueeze(0)

        torch.onnx.export(
            self,
            example_input,
            filename,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

    @staticmethod
    def predict_onnx(filename, features):
        """Load the saved ONNX model and predict from raw feature values."""
        import onnxruntime as ort

        session = ort.InferenceSession(
            filename,
            providers=["CPUExecutionProvider"],
        )
        x = np.atleast_2d(np.asarray(features, dtype=np.float32))
        logits = session.run(["logits"], {"features": x})[0]

        # A logit of zero corresponds to a probability of 0.5.
        return (logits >= 0).astype(int)
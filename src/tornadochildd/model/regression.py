import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Start both weights at zero.
        self.w0 = torch.tensor(
            0.0, dtype=torch.float64, requires_grad=True
        )
        self.w1 = torch.tensor(
            0.0, dtype=torch.float64, requires_grad=True
        )

        self.optimizer = torch.optim.SGD(
            [self.w0, self.w1], lr=self.learning_rate
        )
        self.loss_function = torch.nn.MSELoss()

        # Save the weights and loss after each update.
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []

    def forward(self, x):
        return self.w0 + self.w1 * x

    def fit(self, x_train, y_train, x_test, y_test):
        # Use one-dimensional tensors with matching shapes.
        x_train = torch.as_tensor(
            np.asarray(x_train), dtype=torch.float64
        ).reshape(-1)
        y_train = torch.as_tensor(
            np.asarray(y_train), dtype=torch.float64
        ).reshape(-1)

        if len(x_train) != len(y_train) or len(x_train) == 0:
            raise ValueError("Training inputs and outputs must match.")

        self.x_train = x_train.detach().clone()
        self.y_train = y_train.detach().clone()

        self.w0_history.clear()
        self.w1_history.clear()
        self.loss_history.clear()

        for epoch in range(self.epochs):
            # Clear the previous gradients.
            self.optimizer.zero_grad()

            # Predict and calculate the error.
            y_pred = self.forward(x_train)
            loss = self.loss_function(y_pred, y_train)

            # Calculate gradients and update the weights.
            loss.backward()
            self.optimizer.step()

            # Record the updated weights and their loss.
            with torch.no_grad():
                updated_loss = self.loss_function(
                    self.forward(x_train), y_train
                )

                if not torch.isfinite(updated_loss):
                    raise ValueError(
                        "Training diverged. Try a smaller learning rate."
                    )

                self.w0_history.append(self.w0.item())
                self.w1_history.append(self.w1.item())
                self.loss_history.append(updated_loss.item())

        # Evaluate the trained model on the test data.
        y_test = np.asarray(y_test).reshape(-1)
        test_predictions = self.predict(x_test)
        self.test_r2 = r2_score(y_test, test_predictions)

        print(f"w0 (intercept): {self.w0.item():.6f}")
        print(f"w1 (slope): {self.w1.item():.6f}")
        print(f"Test R²: {self.test_r2:.6f}")

        return self

    def predict(self, x):
        x = torch.as_tensor(
            np.asarray(x), dtype=torch.float64
        ).reshape(-1)

        with torch.no_grad():
            return self.forward(x).numpy()

    def analysis_plot(self):
        if not self.loss_history:
            raise ValueError("Call fit before analysis_plot.")

        x = self.x_train.numpy()
        y = self.y_train.numpy()

        # Sort the input values to draw a smooth line.
        order = np.argsort(x)
        x_sorted = x[order]
        y_fitted = self.predict(x_sorted)

        epochs = np.arange(1, len(self.loss_history) + 1)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        axes[0, 0].scatter(x, y, alpha=0.7)
        axes[0, 0].set_title("Original training data")
        axes[0, 0].set_xlabel("BCR")
        axes[0, 0].set_ylabel("Annual Production")

        axes[0, 1].scatter(x, y, alpha=0.6, label="Training data")
        axes[0, 1].plot(
            x_sorted, y_fitted, color="red", label="Fitted line"
        )
        axes[0, 1].set_title("Fitted regression line")
        axes[0, 1].set_xlabel("BCR")
        axes[0, 1].set_ylabel("Annual Production")
        axes[0, 1].legend()

        axes[0, 2].plot(epochs, self.w0_history)
        axes[0, 2].set_title("Intercept during training")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("w0")

        axes[1, 0].plot(epochs, self.w1_history)
        axes[1, 0].set_title("Slope during training")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("w1")

        axes[1, 1].plot(epochs, self.loss_history)
        axes[1, 1].set_title("Loss during training")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Mean squared error")

        axes[1, 2].axis("off")

        fig.tight_layout()
        plt.show()
        return fig
import numpy as np
import torch


class LogisticRegression:
    def __init__(self, learning_rate=0.001, epochs=50, seed=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights = None
        self.bias = None
        self.train_loss_history = []
        self.val_loss_history = []

    def forward(self, X):
        # Return linear scores (logits).
        return X @ self.weights + self.bias

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = torch.as_tensor(
            np.asarray(X_train), dtype=torch.float64
        )
        y_train = torch.as_tensor(
            np.asarray(y_train), dtype=torch.float64
        ).reshape(-1)

        X_val = torch.as_tensor(
            np.asarray(X_val), dtype=torch.float64
        )
        y_val = torch.as_tensor(
            np.asarray(y_val), dtype=torch.float64
        ).reshape(-1)

        for X, y in [(X_train, y_train), (X_val, y_val)]:
            if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
                raise ValueError("Inputs and labels must have matching rows.")
            if not torch.isfinite(X).all():
                raise ValueError("Features must contain finite values.")
            if not torch.all((y == 0) | (y == 1)):
                raise ValueError("Labels must be 0 or 1.")

        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError("Training and validation features must match.")

        if self.learning_rate <= 0 or self.epochs < 1:
            raise ValueError("Learning rate and epochs must be positive.")

        # Start a fresh fit with zero weights and bias.
        self.weights = torch.zeros(
            X_train.shape[1], dtype=torch.float64, requires_grad=True
        )
        self.bias = torch.tensor(
            0.0, dtype=torch.float64, requires_grad=True
        )

        optimizer = torch.optim.SGD(
            [self.weights, self.bias], lr=self.learning_rate
        )
        loss_function = torch.nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.seed)

        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss_ = float("inf")
        best_weights = None
        best_bias = None

        for epoch in range(self.epochs):
            # Shuffle training observations each epoch.
            order = torch.randperm(
                len(y_train), generator=generator
            ).tolist()

            for i in order:
                optimizer.zero_grad()

                logit = self.forward(X_train[i])
                loss = loss_function(logit, y_train[i])

                loss.backward()
                optimizer.step()

            # Evaluate both losses without updating parameters.
            with torch.no_grad():
                train_loss = loss_function(
                    self.forward(X_train), y_train
                ).item()
                val_loss = loss_function(
                    self.forward(X_val), y_val
                ).item()

            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise ValueError(
                    "Training diverged. Try a smaller learning rate."
                )

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            # Retain the epoch with the lowest validation loss.
            if val_loss < self.best_val_loss_:
                self.best_val_loss_ = val_loss
                self.best_epoch_ = epoch + 1
                best_weights = self.weights.detach().clone()
                best_bias = self.bias.detach().clone()

            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Train loss: {train_loss:.6f} | "
                    f"Validation loss: {val_loss:.6f}"
                )

        with torch.no_grad():
            self.weights.copy_(best_weights)
            self.bias.copy_(best_bias)

        self.coef_ = self.weights.detach().numpy().copy()
        self.intercept_ = self.bias.item()

        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Call fit before making predictions.")

        X = torch.as_tensor(
            np.asarray(X), dtype=torch.float64
        )

        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("Provide a matrix with the fitted feature count.")

        with torch.no_grad():
            return torch.sigmoid(self.forward(X)).numpy()

    def predict(self, X, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")

        return (self.predict_proba(X) >= threshold).astype(int)
"""
Extreme Learning Machine (ELM) classifier.

This is the method used across the project's reference papers
(Sheeba et al. 2022; Suchithra & Pai 2020). A single-hidden-layer
feedforward network whose input weights are random and whose output
weights are solved in one shot with the Moore-Penrose pseudo-inverse
-> extremely fast training, no back-propagation.

Implemented with a scikit-learn-style fit/predict API so it can be
dropped into the same comparison loop as RandomForest / XGBoost.
"""

import numpy as np


def _activation(x, kind):
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    if kind == "tanh":
        return np.tanh(x)
    if kind == "relu":
        return np.maximum(0, x)
    if kind == "grbf":  # gaussian radial basis (best performer in the papers)
        return np.exp(-np.square(x))
    raise ValueError(f"unknown activation: {kind}")


class ELMClassifier:
    def __init__(self, n_hidden=150, activation="sigmoid", random_state=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        rng = np.random.default_rng(self.random_state)
        # random input weights + biases (the ELM idea: never trained)
        self.W_ = rng.normal(size=(n_features, self.n_hidden))
        self.b_ = rng.normal(size=(self.n_hidden,))

        H = _activation(X @ self.W_ + self.b_, self.activation)

        # one-hot targets
        Y = np.zeros((len(y), len(self.classes_)))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for i, label in enumerate(y):
            Y[i, class_to_idx[label]] = 1.0

        # output weights via Moore-Penrose pseudo-inverse
        self.beta_ = np.linalg.pinv(H) @ Y
        return self

    def _scores(self, X):
        X = np.asarray(X, dtype=float)
        H = _activation(X @ self.W_ + self.b_, self.activation)
        return H @ self.beta_

    def predict(self, X):
        scores = self._scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self._scores(X)
        # softmax to turn raw scores into probabilities
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

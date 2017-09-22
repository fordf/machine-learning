import numpy as np


class Normalizer:

    def __init__(self):
        self.fitted = False

    def fit_transform(self, X):
        X_norm = np.zeros_like(X, dtype=np.float64)
        num_features = X.shape[1]
        self.mu = np.zeros(num_features)
        self.sigma = np.zeros(num_features)
        for feature_i in range(num_features):
            column = X[:, feature_i].astype(np.float64)
            mean = np.mean(column)
            stdev = np.std(column)
            self.mu[feature_i] = mean
            self.sigma[feature_i] = stdev
            X_norm[:, feature_i] = (column - mean) / stdev
        self.fitted = True
        return X_norm

    def normalize(self, X):
        if not self.fitted:
            raise Exception('fit_transform first')
        X_norm = np.zeros_like(X)
        for i in range(X.shape[1]):
            column = X[:, i].astype(np.float64)
            X_norm[:, i] = (column - self.mu[i]) / self.sigma[i]
        return X_norm

    def inverse_scale(self, X):
        if not self.fitted:
            raise Exception('fit_transform first')
        X_orig = np.zeros_like(X)
        for i in range(X.shape[1]):
            column = X[:, i]
            X_orig[:, i] = column * self.sigma[i] + self.mu[i]
        return X_orig
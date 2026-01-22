import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # 1. Initialize centroids randomly from the dataset
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # 2. Assign samples to the nearest centroid
            self.labels = self._assign_labels(X)
            
            # 3. Compute new centroids as the mean of assigned samples
            new_centroids = self._compute_centroids(X)
            
            # 4. Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
                
            self.centroids = new_centroids

    def _assign_labels(self, X):
        # Compute Euclidean distance between each sample and each centroid
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids)**2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X):
        centroids = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            centroids[k] = np.mean(X[self.labels == k], axis=0)
        return centroids

    def predict(self, X):
        return self._assign_labels(X)

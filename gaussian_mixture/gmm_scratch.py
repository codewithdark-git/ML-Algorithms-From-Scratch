import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covs = None
        self.responsibilities = None

    def fit(self, X):
        n_samples, n_features = X.shape

        # 1. Initialization
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covs = [np.eye(n_features) for _ in range(self.n_components)]

        for i in range(self.max_iter):
            prev_means = self.means.copy()

            # E-Step: Compute responsibilities
            self.responsibilities = self._expectation(X)

            # M-Step: Update parameters
            self._maximization(X)

            # Check convergence
            if np.all(np.abs(self.means - prev_means) < self.tol):
                break

    def _expectation(self, X):
        n_samples = X.shape[0]
        weighted_pdfs = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            weighted_pdfs[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], cov=self.covs[k])

        return weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

    def _maximization(self, X):
        n_samples, n_features = X.shape
        nk = self.responsibilities.sum(axis=0)

        for k in range(self.n_components):
            # Update mean
            self.means[k] = (self.responsibilities[:, k, np.newaxis] * X).sum(axis=0) / nk[k]
            
            # Update covariance
            diff = X - self.means[k]
            self.covs[k] = np.dot(self.responsibilities[:, k] * diff.T, diff) / nk[k]
            
            # Update weight
            self.weights[k] = nk[k] / n_samples

    def predict(self, X):
        responsibilities = self._expectation(X)
        return np.argmax(responsibilities, axis=1)

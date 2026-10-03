"""Optimization algorithms: Gradient Descent variants and learning rate schedules."""

import numpy as np
from typing import Optional, Literal


def batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iters: int = 1000,
    fit_intercept: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """
    Batch gradient descent for linear regression.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    learning_rate : float, default=0.01
        Step size for weight updates.
    n_iters : int, default=1000
        Number of iterations.
    fit_intercept : bool, default=True
        Whether to add bias column.
    verbose : bool, default=False
        Print progress.
        
    Returns
    -------
    weights : ndarray, shape (n_features + 1,) or (n_features,)
        Learned weights (includes intercept if fit_intercept=True).
    cost_history : list of float
        Cost at each iteration.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    
    if fit_intercept:
        X = np.column_stack((np.ones(X.shape[0]), X))
    
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    cost_history = []
    
    for iteration in range(n_iters):
        predictions = X @ weights
        errors = predictions - y
        
        gradient = X.T @ errors / n_samples
        weights -= learning_rate * gradient
        
        cost = np.mean(errors ** 2) / 2
        cost_history.append(cost)
        
        if verbose and iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost = {cost:.6f}")
    
    return weights, cost_history


def stochastic_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_epochs: int = 100,
    fit_intercept: bool = True,
    shuffle: bool = True,
    verbose: bool = False,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Stochastic gradient descent (one example per update).
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    learning_rate : float, default=0.01
        Step size for weight updates.
    n_epochs : int, default=100
        Number of passes through the data.
    fit_intercept : bool, default=True
        Whether to add bias column.
    shuffle : bool, default=True
        Shuffle data each epoch.
    verbose : bool, default=False
        Print progress.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    weights : ndarray
        Learned weights.
    cost_history : list of float
        Average cost per epoch.
    """
    rng = np.random.default_rng(random_state)
    
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    
    if fit_intercept:
        X = np.column_stack((np.ones(X.shape[0]), X))
    
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    cost_history = []
    
    for epoch in range(n_epochs):
        if shuffle:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        
        epoch_cost = 0.0
        
        for i in indices:
            xi = X[i:i+1]
            yi = y[i]
            
            prediction = (xi @ weights)[0]
            error = prediction - yi
            
            gradient = xi.T @ np.array([[error]])
            weights -= learning_rate * gradient.flatten()
            
            epoch_cost += error ** 2
        
        avg_cost = epoch_cost / (2 * n_samples)
        cost_history.append(avg_cost)
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Average Cost = {avg_cost:.6f}")
    
    return weights, cost_history


def minibatch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_epochs: int = 100,
    batch_size: int = 32,
    fit_intercept: bool = True,
    shuffle: bool = True,
    verbose: bool = False,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Mini-batch gradient descent.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    learning_rate : float, default=0.01
        Step size for weight updates.
    n_epochs : int, default=100
        Number of passes through the data.
    batch_size : int, default=32
        Number of samples per mini-batch.
    fit_intercept : bool, default=True
        Whether to add bias column.
    shuffle : bool, default=True
        Shuffle data each epoch.
    verbose : bool, default=False
        Print progress.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    weights : ndarray
        Learned weights.
    cost_history : list of float
        Average cost per epoch.
    """
    rng = np.random.default_rng(random_state)
    
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    
    if fit_intercept:
        X = np.column_stack((np.ones(X.shape[0]), X))
    
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    cost_history = []
    
    for epoch in range(n_epochs):
        if shuffle:
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        
        epoch_cost = 0.0
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            predictions = X_batch @ weights
            errors = predictions - y_batch
            
            gradient = X_batch.T @ errors / len(batch_indices)
            weights -= learning_rate * gradient
            
            epoch_cost += np.sum(errors ** 2)
        
        avg_cost = epoch_cost / (2 * n_samples)
        cost_history.append(avg_cost)
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Average Cost = {avg_cost:.6f}")
    
    return weights, cost_history


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iters: int = 1000,
    batch_size: Optional[int] = None,
    fit_intercept: bool = True,
    shuffle: bool = True,
    verbose: bool = False,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Unified gradient descent function.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    learning_rate : float, default=0.01
        Step size.
    n_iters : int, default=1000
        Iterations (for batch) or epochs (for SGD/minibatch).
    batch_size : int or None, default=None
        - None or n_samples: batch gradient descent
        - 1: stochastic gradient descent
        - Other: mini-batch gradient descent
    fit_intercept : bool, default=True
        Whether to add bias column.
    shuffle : bool, default=True
        Shuffle data each epoch (for SGD/minibatch).
    verbose : bool, default=False
        Print progress.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    weights : ndarray
        Learned weights.
    cost_history : list of float
        Cost history.
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    
    if batch_size is None or batch_size >= n_samples:
        return batch_gradient_descent(
            X, y, learning_rate, n_iters, fit_intercept, verbose
        )
    elif batch_size == 1:
        return stochastic_gradient_descent(
            X, y, learning_rate, n_iters, fit_intercept, shuffle, verbose, random_state
        )
    else:
        return minibatch_gradient_descent(
            X, y, learning_rate, n_iters, batch_size, fit_intercept, shuffle, verbose, random_state
        )


def learning_rate_schedule(
    initial_lr: float,
    schedule: Literal["constant", "step", "exponential", "inverse_time", "cosine"] = "constant",
    epoch: int = 0,
    total_epochs: int = 100,
    decay_rate: float = 0.1,
    decay_steps: int = 10,
) -> float:
    """
    Learning rate schedules.
    
    Parameters
    ----------
    initial_lr : float
        Initial learning rate.
    schedule : str, default="constant"
        Schedule type: "constant", "step", "exponential", "inverse_time", "cosine".
    epoch : int, default=0
        Current epoch/iteration.
    total_epochs : int, default=100
        Total epochs (for cosine annealing).
    decay_rate : float, default=0.1
        Decay factor.
    decay_steps : int, default=10
        Steps between decays (for step decay).
        
    Returns
    -------
    lr : float
        Adjusted learning rate.
    """
    if schedule == "constant":
        return initial_lr
    elif schedule == "step":
        return initial_lr * (decay_rate ** (epoch // decay_steps))
    elif schedule == "exponential":
        return initial_lr * np.exp(-decay_rate * epoch)
    elif schedule == "inverse_time":
        return initial_lr / (1 + decay_rate * epoch)
    elif schedule == "cosine":
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class MomentumOptimizer:
    """SGD with momentum."""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        n_epochs: int = 100,
        batch_size: int = 32,
        fit_intercept: bool = True,
        shuffle: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self.velocity = None
        self.weights = None
        self.cost_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MomentumOptimizer":
        rng = np.random.default_rng(self.random_state)
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        
        if self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.velocity = np.zeros(n_features)
        self.cost_history = []
        
        for epoch in range(self.n_epochs):
            if self.shuffle:
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)
            
            epoch_cost = 0.0
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                predictions = X_batch @ self.weights
                errors = predictions - y_batch
                
                gradient = X_batch.T @ errors / len(batch_indices)
                
                # Momentum update
                self.velocity = self.momentum * self.velocity + gradient
                self.weights -= self.learning_rate * self.velocity
                
                epoch_cost += np.sum(errors ** 2)
            
            avg_cost = epoch_cost / (2 * n_samples)
            self.cost_history.append(avg_cost)
            
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Average Cost = {avg_cost:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        return X @ self.weights
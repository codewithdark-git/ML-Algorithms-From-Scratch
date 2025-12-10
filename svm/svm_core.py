import numpy as np

class SVM:
    def __init__(self, kernel='linear', C=1.0, max_iter=1000, tol=1e-3, gamma=0.1, degree=3, coef0=1, record_history=False):
        """
        Support Vector Machine using Sequential Minimal Optimization (SMO).
        
        Parameters:
        - kernel: 'linear', 'poly', or 'rbf'
        - C: Regularization parameter (Soft margin). Large C -> Hard Margin.
        - max_iter: Maximum passes without change
        - tol: Numerical tolerance
        - gamma: Kernel coefficient for 'rbf' and 'poly'
        - degree: Degree for 'poly' kernel
        - coef0: Independent term in kernel function for 'poly'
        - record_history: Boolean, whether to store parameters at each step for visualization
        """
        self.kernel_type = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.record_history = record_history
        
        self.history = []
        self.alpha = None
        self.b = 0
        self.w = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.X = None
        self.y = None

    def _kernel(self, x1, x2):
        if self.kernel_type == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel_type == 'poly':
            return (self.gamma * np.dot(x1, x2.T) + self.coef0) ** self.degree
        elif self.kernel_type == 'rbf':
            # Efficient RBF computation using (a-b)^2 = a^2 + b^2 - 2ab
            if x1.ndim == 1 and x2.ndim == 1:
                return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
            elif (x1.ndim > 1 and x2.ndim == 1) or (x1.ndim == 1 and x2.ndim > 1):
                 return np.exp(-self.gamma * np.linalg.norm(x1 - x2, axis=-1) ** 2)
            else:
                # Pairwise distance matrix calculation could be optimized but sticking to basic for clarity
                # or utilizing broadcasting if memory allows. 
                # For simplicity in 'from scratch', we can iterate or use scipy cdist-like logic manually
                # But let's use a broadcasting trick for small to med datasets
                dists = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
                return np.exp(-self.gamma * dists)
        return np.dot(x1, x2.T)

    def fit(self, X, y):
        """
        Train the SVM model using Simplified SMO.
        """
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.history = []
        
        passes = 0
        while passes < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):
                # Calculate Ei = f(xi) - yi
                # f(x) = sum(alpha_j * y_j * K(x_j, x)) + b
                # We can precompute specific kernel rows if needed, but for simplicity:
                prediction_i = np.sum(self.alpha * self.y * self._kernel(self.X, self.X[i])) + self.b
                E_i = prediction_i - self.y[i]
                
                # Check KKT conditions
                if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (self.y[i] * E_i > self.tol and self.alpha[i] > 0):
                    
                    # Select j randomly (Simplified SMO)
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                        
                    prediction_j = np.sum(self.alpha * self.y * self._kernel(self.X, self.X[j])) + self.b
                    E_j = prediction_j - self.y[j]
                    
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute L and H
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                        
                    if L == H:
                        continue
                        
                    # Compute eta
                    # eta = 2 * K(i, j) - K(i, i) - K(j, j)
                    # Note: _kernel checks shape, so we pass single vectors or handle it
                    k_ij = self._kernel(self.X[i].reshape(1, -1), self.X[j].reshape(1, -1)).item()
                    k_ii = self._kernel(self.X[i].reshape(1, -1), self.X[i].reshape(1, -1)).item()
                    k_jj = self._kernel(self.X[j].reshape(1, -1), self.X[j].reshape(1, -1)).item()
                    
                    eta = 2 * k_ij - k_ii - k_jj
                    
                    if eta >= 0:
                        continue
                        
                    # Update alpha_j
                    self.alpha[j] -= self.y[j] * (E_i - E_j) / eta
                    
                    # Clip alpha_j
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L
                        
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                        
                    # Update alpha_i
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update b
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * k_ii - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * k_ij
                    b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * k_ij - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * k_jj
                         
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                        
                    num_changed_alphas += 1
                    
                    if self.record_history: # Record every update for smooth animations
                         self.history.append({
                            'alpha': self.alpha.copy(),
                            'b': self.b,
                            'w': np.dot((self.alpha * self.y).T, self.X) if self.kernel_type == 'linear' else None
                        })
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                
        # Store support vectors
        self.sv_indices = np.where(self.alpha > 0)[0]
        self.support_vectors = self.X[self.sv_indices]
        self.support_vector_labels = self.y[self.sv_indices]
        self.alpha = self.alpha[self.sv_indices]
        
        # Calculate w if linear kernel
        if self.kernel_type == 'linear':
            self.w = np.dot((self.alpha * self.support_vector_labels).T, self.support_vectors)
        else:
            self.w = None
            
    def predict(self, X):
        if self.w is not None and self.kernel_type == 'linear':
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for alpha, sv_y, sv in zip(self.alpha, self.support_vector_labels, self.support_vectors):
                    s += alpha * sv_y * self._kernel(X[i], sv)
                y_predict[i] = s
            return np.sign(y_predict + self.b)

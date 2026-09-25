"""Plotting utilities for ML from Scratch - beginner-friendly visualizations."""

import numpy as np
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def plot_learning_curve(
    cost_history: List[float],
    title: str = "Learning Curve",
    xlabel: str = "Iteration",
    ylabel: str = "Cost",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot the learning curve (cost vs iterations).
    
    Parameters
    ----------
    cost_history : list of float
        Cost values at each iteration.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
        
    Returns
    -------
    fig : matplotlib.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(cost_history, 'b-', linewidth=1.5, label='Training Cost')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add annotations
    if len(cost_history) > 0:
        ax.annotate(f'Initial: {cost_history[0]:.4f}', 
                   xy=(0, cost_history[0]), xytext=(10, 10),
                   textcoords='offset points', fontsize=10, color='red')
        ax.annotate(f'Final: {cost_history[-1]:.4f}', 
                   xy=(len(cost_history)-1, cost_history[-1]), xytext=(-10, -15),
                   textcoords='offset points', fontsize=10, color='green')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Learning curve saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    resolution: float = 0.02,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot decision boundary for 2D classification.
    
    Parameters
    ----------
    model : fitted estimator
        Model with predict method.
    X : array-like, shape (n_samples, 2)
        Training features (must be 2D).
    y : array-like, shape (n_samples,)
        Training labels.
    title : str
        Plot title.
    resolution : float
        Mesh grid resolution.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display plot.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plot requires 2D features (X.shape[1] == 2)")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define the mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
    
    # Plot training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                        edgecolors='k', s=50, alpha=0.8)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Class')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Decision boundary saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_regression_line(
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Linear Regression Fit",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot regression line for 1D regression.
    
    Parameters
    ----------
    model : fitted estimator
        Model with predict method.
    X : array-like, shape (n_samples, 1) or (n_samples,)
        Training features.
    y : array-like, shape (n_samples,)
        Training targets.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display plot.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot training data
    ax.scatter(X, y, alpha=0.6, s=50, label='Data', color='steelblue', edgecolors='k')
    
    # Plot regression line
    X_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
    X_line_2d = X_line.reshape(-1, 1)
    y_line = model.predict(X_line_2d)
    ax.plot(X_line, y_line, 'r-', linewidth=2, label='Prediction')
    
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Regression plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_cost_comparison(
    cost_histories: dict,
    title: str = "Cost Comparison",
    xlabel: str = "Iteration",
    ylabel: str = "Cost",
    log_scale: bool = False,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Compare multiple cost histories (e.g., different learning rates).
    
    Parameters
    ----------
    cost_histories : dict
        Dictionary mapping label -> cost_history list.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    log_scale : bool
        Use log scale for y-axis.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display plot.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['steelblue', 'coral', 'seagreen', 'gold', 'mediumpurple', 'tomato']
    
    for i, (label, history) in enumerate(cost_histories.items()):
        color = colors[i % len(colors)]
        ax.plot(history, label=label, color=color, linewidth=1.5, alpha=0.8)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Cost comparison saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_feature_importance(
    features: List[str],
    importance: np.ndarray,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    features : list of str
        Feature names.
    importance : array-like
        Importance values.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display plot.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.4)))
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    features_sorted = [features[i] for i in sorted_idx]
    importance_sorted = importance[sorted_idx]
    
    bars = ax.barh(range(len(features_sorted)), importance_sorted, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted, fontsize=11)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance_sorted)):
        ax.text(val + 0.01 * max(importance_sorted), bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Parameters
    ----------
    cm : array-like, shape (n_classes, n_classes)
        Confusion matrix.
    classes : list of str
        Class names.
    title : str
        Plot title.
    normalize : bool
        Normalize by row (true class).
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display plot.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    silhouette_scores: np.ndarray,
    n_clusters: int,
    title: str = "Silhouette Analysis",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot silhouette analysis for clustering.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data.
    labels : array-like, shape (n_samples,)
        Cluster labels.
    silhouette_scores : array-like, shape (n_samples,)
        Silhouette scores per sample.
    n_clusters : int
        Number of clusters.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display plot.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    y_lower = 10
    for i in range(n_clusters):
        cluster_scores = silhouette_scores[labels == i]
        cluster_scores.sort()
        size = cluster_scores.shape[0]
        y_upper = y_lower + size
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_scores,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size, str(i), fontsize=12, fontweight='bold')
        y_lower = y_upper + 10
    
    avg_score = np.mean(silhouette_scores)
    ax1.axvline(x=avg_score, color="red", linestyle="--", linewidth=2, 
                label=f"Average: {avg_score:.3f}")
    ax1.set_xlabel("Silhouette Score", fontsize=12)
    ax1.set_ylabel("Cluster", fontsize=12)
    ax1.set_title("Silhouette Plot", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xlim([-0.1, 1])
    
    # 2D scatter of data colored by cluster
    if X.shape[1] >= 2:
        scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.nipy_spectral, 
                            s=30, alpha=0.7, edgecolors='k')
        ax2.set_xlabel('Feature 1', fontsize=12)
        ax2.set_ylabel('Feature 2', fontsize=12)
        ax2.set_title("Data colored by Cluster", fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax2)
    else:
        ax2.text(0.5, 0.5, "Need 2D+ data for scatter", ha='center', va='center', fontsize=14)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Silhouette analysis saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_elbow_curve(
    k_range: List[int],
    inertias: List[float],
    title: str = "Elbow Method for Optimal K",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot elbow curve for K-Means.
    
    Parameters
    ----------
    k_range : list of int
        K values tested.
    inertias : list of float
        Inertia for each K.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display plot.
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_range)
    
    # Mark the elbow if possible (largest curvature)
    if len(k_range) >= 3:
        inertias_arr = np.array(inertias)
        # Second derivative approximation
        diff1 = np.diff(inertias_arr)
        diff2 = np.diff(diff1)
        if len(diff2) > 0:
            elbow_idx = np.argmax(np.abs(diff2)) + 2  # +2 because of two diffs
            if elbow_idx < len(k_range):
                ax.annotate(f'Elbow at K={k_range[elbow_idx]}',
                           xy=(k_range[elbow_idx], inertias[elbow_idx]),
                           xytext=(k_range[elbow_idx] + 0.5, inertias[elbow_idx]),
                           fontsize=11, color='red', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Elbow curve saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
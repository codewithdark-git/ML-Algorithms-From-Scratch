import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

class SVMVisualizer:
    """
    Enhanced SVM Visualizer for educational purposes.
    Creates clear, annotated animations showing how SVM finds the optimal hyperplane.
    """
    
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
        # Define a nice color scheme
        self.colors = {
            'class_pos': '#FF6B6B',    # Coral red for class +1
            'class_neg': '#4ECDC4',    # Teal for class -1
            'boundary': '#2C3E50',      # Dark blue-gray for decision boundary
            'margin': '#95A5A6',        # Gray for margin lines
            'support_vec': '#F39C12',   # Orange for support vectors
            'background_pos': '#FFEBEE',
            'background_neg': '#E0F7FA'
        }

    def plot_decision_boundary(self, title="SVM Decision Boundary", save_path=None):
        """Plot the final decision boundary with support vectors highlighted."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up the meshgrid for background coloring
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        xy = np.c_[xx.ravel(), yy.ravel()]
        
        # Get decision function values
        Z = self._decision_function(xy).reshape(xx.shape)
        
        # Plot decision regions with soft colors
        ax.contourf(xx, yy, Z, levels=[-100, 0, 100], 
                    colors=[self.colors['background_neg'], self.colors['background_pos']], alpha=0.4)
        
        # Plot decision boundary (Z=0), and margins (Z=-1, Z=1)
        contours = ax.contour(xx, yy, Z, levels=[-1, 0, 1], 
                              colors=[self.colors['margin'], self.colors['boundary'], self.colors['margin']],
                              linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])
        ax.clabel(contours, fmt={-1: 'Margin -1', 0: 'Decision Boundary', 1: 'Margin +1'}, fontsize=9)
        
        # Plot data points
        pos_mask = self.y == 1
        neg_mask = self.y == -1
        ax.scatter(self.X[pos_mask, 0], self.X[pos_mask, 1], 
                   c=self.colors['class_pos'], s=80, edgecolors='k', linewidth=1, label='Class +1', zorder=5)
        ax.scatter(self.X[neg_mask, 0], self.X[neg_mask, 1], 
                   c=self.colors['class_neg'], s=80, edgecolors='k', linewidth=1, label='Class -1', zorder=5)
        
        # Highlight support vectors
        if self.model.support_vectors is not None and len(self.model.support_vectors) > 0:
            ax.scatter(self.model.support_vectors[:, 0], self.model.support_vectors[:, 1],
                       s=200, linewidth=2.5, facecolors='none', 
                       edgecolors=self.colors['support_vec'], label='Support Vectors', zorder=6)
        
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()
        return fig

    def create_educational_gif(self, filename='svm_training.gif', fps=5):
        """
        Create an enhanced educational GIF showing how SVM finds the optimal margin.
        Shows:
        1. The decision boundary moving
        2. The margins adjusting
        3. Support vectors being identified
        4. Annotations explaining the process
        """
        if not self.model.history or len(self.model.history) == 0:
            print("No history found. Please train the model with record_history=True")
            return
        
        # Select frames to show (sample if too many, to keep GIF manageable)
        total_frames = len(self.model.history)
        if total_frames > 60:
            frame_indices = np.linspace(0, total_frames - 1, 60, dtype=int)
        else:
            frame_indices = list(range(total_frames))
        
        # Add extra frames at the end to pause on final result
        frame_indices = list(frame_indices) + [total_frames - 1] * 10
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up the meshgrid
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))
        xy = np.c_[xx.ravel(), yy.ravel()]
        
        def update(frame_idx):
            ax.clear()
            
            # Get actual frame index
            actual_frame = frame_indices[frame_idx]
            state = self.model.history[actual_frame]
            alpha = state['alpha']
            b = state['b']
            
            # Compute decision function for this frame
            Z = self._compute_decision_for_frame(xy, alpha, b, state.get('w'))
            Z = Z.reshape(xx.shape)
            
            # Plot decision regions
            try:
                ax.contourf(xx, yy, Z, levels=[-100, 0, 100], 
                           colors=[self.colors['background_neg'], self.colors['background_pos']], alpha=0.3)
            except:
                pass
            
            # Plot decision boundary and margins
            try:
                contours = ax.contour(xx, yy, Z, levels=[-1, 0, 1], 
                                      colors=[self.colors['margin'], self.colors['boundary'], self.colors['margin']],
                                      linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])
            except:
                pass
            
            # Plot data points
            pos_mask = self.y == 1
            neg_mask = self.y == -1
            ax.scatter(self.X[pos_mask, 0], self.X[pos_mask, 1], 
                       c=self.colors['class_pos'], s=80, edgecolors='k', linewidth=1, label='Class +1', zorder=5)
            ax.scatter(self.X[neg_mask, 0], self.X[neg_mask, 1], 
                       c=self.colors['class_neg'], s=80, edgecolors='k', linewidth=1, label='Class -1', zorder=5)
            
            # Highlight current support vectors (points with alpha > 0)
            sv_indices = np.where(alpha > 1e-5)[0]
            if len(sv_indices) > 0:
                ax.scatter(self.X[sv_indices, 0], self.X[sv_indices, 1],
                           s=200, linewidth=2.5, facecolors='none', 
                           edgecolors=self.colors['support_vec'], label=f'Support Vectors ({len(sv_indices)})', zorder=6)
            
            # Calculate margin width (for linear kernel)
            margin_width = "N/A"
            if state.get('w') is not None:
                w = state['w']
                w_norm = np.linalg.norm(w)
                if w_norm > 0:
                    margin_width = f"{2/w_norm:.3f}"
            
            # Progress info
            progress = (actual_frame + 1) / total_frames * 100
            
            # Title with training info
            ax.set_title(f'SVM Training Progress\n'
                        f'Step {actual_frame + 1}/{total_frames} ({progress:.0f}%) | '
                        f'Margin Width: {margin_width}', 
                        fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Feature 1', fontsize=11)
            ax.set_ylabel('Feature 2', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Add annotation box explaining what's happening
            if actual_frame < total_frames * 0.1:
                annotation = "🔍 Starting: Initializing hyperplane..."
            elif actual_frame < total_frames * 0.3:
                annotation = "⚙️ Optimizing: Adjusting decision boundary..."
            elif actual_frame < total_frames * 0.6:
                annotation = "📐 Refining: Maximizing margin width..."
            elif actual_frame < total_frames * 0.9:
                annotation = "🎯 Fine-tuning: Finding support vectors..."
            else:
                annotation = "✅ Converged: Optimal hyperplane found!"
            
            ax.text(0.02, 0.98, annotation, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=1000//fps, repeat=True)
        ani.save(filename, writer='pillow', fps=fps)
        plt.close()
        print(f"Saved animation to {filename}")

    def create_margin_evolution_gif(self, filename='margin_evolution.gif'):
        """
        Create a focused GIF showing how the margin evolves during training.
        Best for understanding the concept of margin maximization.
        """
        if not self.model.history or len(self.model.history) == 0:
            print("No history found. Please train the model with record_history=True")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Calculate margin widths for all frames
        margin_widths = []
        for state in self.model.history:
            if state.get('w') is not None:
                w_norm = np.linalg.norm(state['w'])
                if w_norm > 0:
                    margin_widths.append(2/w_norm)
                else:
                    margin_widths.append(0)
            else:
                margin_widths.append(0)
        
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))
        xy = np.c_[xx.ravel(), yy.ravel()]
        
        # Sample frames
        total_frames = len(self.model.history)
        if total_frames > 50:
            frame_indices = np.linspace(0, total_frames - 1, 50, dtype=int)
        else:
            frame_indices = list(range(total_frames))
        frame_indices = list(frame_indices) + [total_frames - 1] * 8
        
        def update(frame_idx):
            ax1.clear()
            ax2.clear()
            
            actual_frame = frame_indices[frame_idx]
            state = self.model.history[actual_frame]
            alpha = state['alpha']
            b = state['b']
            
            # Left plot: Decision boundary
            Z = self._compute_decision_for_frame(xy, alpha, b, state.get('w'))
            Z = Z.reshape(xx.shape)
            
            try:
                ax1.contourf(xx, yy, Z, levels=[-100, 0, 100], 
                            colors=[self.colors['background_neg'], self.colors['background_pos']], alpha=0.3)
                ax1.contour(xx, yy, Z, levels=[-1, 0, 1], 
                           colors=[self.colors['margin'], self.colors['boundary'], self.colors['margin']],
                           linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])
            except:
                pass
            
            # Plot points
            pos_mask = self.y == 1
            neg_mask = self.y == -1
            ax1.scatter(self.X[pos_mask, 0], self.X[pos_mask, 1], 
                        c=self.colors['class_pos'], s=80, edgecolors='k', linewidth=1, zorder=5)
            ax1.scatter(self.X[neg_mask, 0], self.X[neg_mask, 1], 
                        c=self.colors['class_neg'], s=80, edgecolors='k', linewidth=1, zorder=5)
            
            sv_indices = np.where(alpha > 1e-5)[0]
            if len(sv_indices) > 0:
                ax1.scatter(self.X[sv_indices, 0], self.X[sv_indices, 1],
                            s=200, linewidth=2.5, facecolors='none', 
                            edgecolors=self.colors['support_vec'], zorder=6)
            
            ax1.set_title(f'Decision Boundary (Step {actual_frame + 1})', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            
            # Right plot: Margin evolution
            ax2.fill_between(range(actual_frame + 1), margin_widths[:actual_frame + 1], alpha=0.3, color='#3498DB')
            ax2.plot(range(actual_frame + 1), margin_widths[:actual_frame + 1], 
                     color='#2980B9', linewidth=2, marker='o', markersize=3)
            ax2.axhline(y=margin_widths[-1], color='green', linestyle='--', alpha=0.7, label=f'Final: {margin_widths[-1]:.3f}')
            ax2.scatter([actual_frame], [margin_widths[actual_frame]], 
                        color='red', s=100, zorder=10, label='Current')
            
            ax2.set_title('Margin Width Over Training', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Margin Width (2/||w||)')
            ax2.set_xlim(-1, total_frames + 1)
            ax2.set_ylim(0, max(margin_widths) * 1.2 if max(margin_widths) > 0 else 1)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=150)
        ani.save(filename, writer='pillow')
        plt.close()
        print(f"Saved margin evolution animation to {filename}")

    def _compute_decision_for_frame(self, xy, alpha, b, w=None):
        """Compute decision function values for a specific training frame."""
        if self.model.kernel_type == 'linear' and w is not None:
            return np.dot(xy, w) + b
        else:
            sv_indices = np.where(alpha > 1e-5)[0]
            if len(sv_indices) == 0:
                return np.full(len(xy), b)
            
            frame_alpha = alpha[sv_indices]
            frame_sv = self.X[sv_indices]
            frame_sv_y = self.y[sv_indices]
            
            Z = np.zeros(len(xy))
            for i in range(len(xy)):
                s = 0
                for a, sy, sv in zip(frame_alpha, frame_sv_y, frame_sv):
                    s += a * sy * self.model._kernel(xy[i], sv)
                Z[i] = s + b
            return Z

    def _decision_function(self, X):
        """Compute decision function for trained model."""
        if self.model.w is not None and self.model.kernel_type == 'linear':
            return np.dot(X, self.model.w) + self.model.b
        else:
            if self.model.support_vectors is None or len(self.model.support_vectors) == 0:
                return np.zeros(len(X))
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for alpha, sv_y, sv in zip(self.model.alpha, self.model.support_vector_labels, self.model.support_vectors):
                    s += alpha * sv_y * self.model._kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.model.b

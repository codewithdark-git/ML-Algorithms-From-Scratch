import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sys
import os

# Import our custom implementation
from svm_hard import HardMarginSVM
from svm_soft import SoftMarginSVM
from visualizer import SVMVisualizer

def run_comparison():
    print("=" * 60)
    print("        SVM FROM SCRATCH VS SKLEARN COMPARISON")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Hard Margin / Linear Separable
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[TEST 1] Hard Margin SVM (Linear Data)")
    print("-" * 60)
    
    # Generate well-separated linearly separable data
    X, y = make_blobs(n_samples=80, centers=2, random_state=42, cluster_std=1.0)
    y_scratch = np.where(y == 0, -1, 1)
    
    # --- Custom Implementation ---
    print("\n🔧 Training Custom HardMarginSVM...")
    custom_model = HardMarginSVM(kernel='linear', record_history=True, max_iter=500)
    custom_model.fit(X, y_scratch)
    custom_preds = custom_model.predict(X)
    custom_acc = accuracy_score(y_scratch, custom_preds)
    print(f"   ✅ Custom Implementation Accuracy: {custom_acc*100:.2f}%")
    print(f"   📌 Number of Support Vectors: {len(custom_model.support_vectors) if custom_model.support_vectors is not None else 0}")
    print(f"   📊 Training History Frames: {len(custom_model.history)}")
    
    # Visualizations
    viz_hard = SVMVisualizer(custom_model, X, y_scratch)
    
    print("\n📈 Generating Hard Margin visualizations...")
    viz_hard.plot_decision_boundary(title="Hard Margin SVM - Final Decision Boundary", 
                                      save_path="outputs/hard_margin_final.png")
    print("   ✅ Saved: outputs/hard_margin_final.png")
    
    viz_hard.create_educational_gif(filename="outputs/hard_margin_training.gif", fps=8)
    print("   ✅ Saved: outputs/hard_margin_training.gif")
    
    viz_hard.create_margin_evolution_gif(filename="outputs/hard_margin_evolution.gif")
    print("   ✅ Saved: outputs/hard_margin_evolution.gif")
    
    # --- Sklearn Implementation ---
    print("\n🔧 Training Sklearn SVC (Hard Margin)...")
    sk_model = SVC(kernel='linear', C=1e10)
    sk_model.fit(X, y_scratch)
    sk_preds = sk_model.predict(X)
    sk_acc = accuracy_score(y_scratch, sk_preds)
    print(f"   ✅ Sklearn Implementation Accuracy: {sk_acc*100:.2f}%")
    
    if custom_acc >= sk_acc * 0.95:
        print("\n✅ MATCH: Custom implementation matches Sklearn performance!")
    else:
        print("\n⚠️ MISMATCH: Performance deviation detected.")

    # -------------------------------------------------------------
    # 2. Soft Margin / Non-Linear (RBF Kernel)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[TEST 2] Soft Margin SVM (Non-Linear Data - Circles)")
    print("-" * 60)
    
    X_circle, y_circle = make_circles(n_samples=100, noise=0.08, factor=0.5, random_state=42)
    y_circle_scratch = np.where(y_circle == 0, -1, 1)
    
    # --- Custom Implementation ---
    print("\n🔧 Training Custom SoftMarginSVM (RBF Kernel)...")
    gamma_val = 1.0
    custom_soft = SoftMarginSVM(C=10.0, kernel='rbf', gamma=gamma_val, record_history=True, max_iter=500)
    custom_soft.fit(X_circle, y_circle_scratch)
    custom_soft_preds = custom_soft.predict(X_circle)
    custom_soft_acc = accuracy_score(y_circle_scratch, custom_soft_preds)
    print(f"   ✅ Custom Implementation Accuracy: {custom_soft_acc*100:.2f}%")
    print(f"   📌 Number of Support Vectors: {len(custom_soft.support_vectors) if custom_soft.support_vectors is not None else 0}")
    print(f"   📊 Training History Frames: {len(custom_soft.history)}")
    
    viz_soft = SVMVisualizer(custom_soft, X_circle, y_circle_scratch)
    
    print("\n📈 Generating Soft Margin visualizations...")
    viz_soft.plot_decision_boundary(title="Soft Margin SVM (RBF) - Final Decision Boundary", 
                                      save_path="outputs/soft_margin_final.png")
    print("   ✅ Saved: outputs/soft_margin_final.png")
    
    viz_soft.create_educational_gif(filename="outputs/soft_margin_training.gif", fps=8)
    print("   ✅ Saved: outputs/soft_margin_training.gif")
    
    # --- Sklearn Implementation ---
    print("\n🔧 Training Sklearn SVC (RBF)...")
    sk_soft = SVC(kernel='rbf', C=10.0, gamma=gamma_val)
    sk_soft.fit(X_circle, y_circle_scratch)
    sk_soft_preds = sk_soft.predict(X_circle)
    sk_soft_acc = accuracy_score(y_circle_scratch, sk_soft_preds)
    print(f"   ✅ Sklearn Implementation Accuracy: {sk_soft_acc*100:.2f}%")
    
    if custom_soft_acc >= sk_soft_acc * 0.9:
        print("\n✅ MATCH: Custom implementation matches Sklearn performance!")
    else:
        print("\n⚠️ MISMATCH: Performance deviation detected.")

    # Summary
    print("\n" + "=" * 60)
    print("                       SUMMARY")
    print("=" * 60)
    print(f"  Hard Margin SVM:  Custom={custom_acc*100:.1f}%  vs  Sklearn={sk_acc*100:.1f}%")
    print(f"  Soft Margin SVM:  Custom={custom_soft_acc*100:.1f}%  vs  Sklearn={sk_soft_acc*100:.1f}%")
    print("\n📁 All outputs saved in: svm/outputs/")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()

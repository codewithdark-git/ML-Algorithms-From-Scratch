#!/usr/bin/env python
"""
Production ML pipeline demo for expert learners.

Usage:
    python scripts/production_demo.py --config config/production.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, "src")

from ml_from_scratch import (
    LogisticRegression, StandardScaler, train_test_split,
    cross_val_score, accuracy_score,
    make_classification,
)
from ml_from_scratch.pipeline import Pipeline
from ml_from_scratch.production import ModelMonitor, ModelSerializer, ExperimentTracker


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_sample_config() -> dict:
    """Create a sample production config."""
    return {
        "data": {
            "source": "synthetic",
            "n_samples": 10000,
            "n_features": 50,
            "test_size": 0.2,
            "random_state": 42,
        },
        "preprocessing": {
            "scaler": "StandardScaler",
            "imputer": None,
        },
        "model": {
            "type": "LogisticRegression",
            "params": {
                "max_iter": 1000,
                "learning_rate": 0.01,
                "verbose": True,
            },
        },
        "training": {
            "cv_folds": 5,
            "stratified": True,
            "scoring": "accuracy",
        },
        "validation": {
            "threshold": 0.5,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
        "monitoring": {
            "drift_threshold": 0.1,
            "performance_threshold": 0.05,
            "reference_window": 1000,
        },
        "deployment": {
            "serialization": "joblib",
            "version": "1.0.0",
        },
    }


def run_pipeline(config: dict) -> dict:
    """Run the full production pipeline."""
    results = {}
    
    print("🚀 PRODUCTION ML PIPELINE DEMO")
    print("=" * 60)
    
    # 1. Data Loading
    print("\n📊 1. DATA LOADING")
    print("-" * 40)
    data_cfg = config["data"]
    if data_cfg["source"] == "synthetic":
        X, y = make_classification(
            n_samples=data_cfg["n_samples"],
            n_features=data_cfg["n_features"],
            n_classes=2,
            n_informative=data_cfg["n_features"] // 2,
            random_state=data_cfg["random_state"],
        )
    print(f"   Loaded: X={X.shape}, y={y.shape}")
    results["data_shape"] = X.shape
    
    # 2. Train/Test Split
    print("\n2. TRAIN/TEST SPLIT")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    results["split"] = {"train": X_train.shape, "test": X_test.shape}
    
    # 3. Preprocessing
    print("\n🔧 3. PREPROCESSING")
    print("-" * 40)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   Scaled features (mean≈0, std≈1)")
    results["preprocessing"] = "StandardScaler"
    
    # 4. Model Training
    print("\n🤖 4. MODEL TRAINING")
    print("-" * 40)
    model_cfg = config["model"]
    model = LogisticRegression(**model_cfg["params"])
    
    # Cross-validation
    train_cfg = config["training"]
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=train_cfg["cv_folds"],
        scoring=train_cfg["scoring"],
    )
    print(f"   CV Scores: {cv_scores}")
    print(f"   Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    results["cv_scores"] = cv_scores.tolist()
    results["cv_mean"] = float(cv_scores.mean())
    results["cv_std"] = float(cv_scores.std())
    
    # Final training
    model.fit(X_train_scaled, y_train)
    print(f"   Final model trained")
    results["model_trained"] = True
    
    # 5. Evaluation
    print("\n📈 5. EVALUATION")
    print("-" * 40)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    from ml_from_scratch.metrics import precision_score, recall_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1:        {f1:.4f}")
    results["test_metrics"] = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    }
    
    # 6. Monitoring Setup
    print("\n🔍 6. MONITORING SETUP")
    print("-" * 40)
    monitor_cfg = config["monitoring"]
    monitor = ModelMonitor(
        model=model,
        reference_data=X_train_scaled[:monitor_cfg["reference_window"]],
        drift_threshold=monitor_cfg["drift_threshold"],
    )
    print(f"   ModelMonitor initialized")
    print(f"   Drift threshold: {monitor_cfg['drift_threshold']}")
    results["monitoring"] = "ModelMonitor initialized"
    
    # Simulate drift detection
    print("\n   Simulating prediction logging...")
    for i in range(5):
        idx = i * 100
        monitor.log_prediction(X_test_scaled[idx:idx+1], y_pred[idx:idx+1])
    alert = monitor.check_drift()
    print(f"   Drift check: {'⚠️ ALERT' if alert else 'OK'}")
    results["drift_alert"] = bool(alert)
    
    # 7. Serialization
    print("\n💾 7. MODEL SERIALIZATION")
    print("-" * 40)
    serializer = ModelSerializer()
    deploy_cfg = config["deployment"]
    model_path = f"models/production_model_v{deploy_cfg['version']}.pkl"
    Path("models").mkdir(exist_ok=True)
    serializer.save(model, model_path)
    print(f"   Saved to: {model_path}")
    results["model_path"] = model_path
    
    # Verify load
    loaded_model = serializer.load(model_path)
    y_pred_loaded = loaded_model.predict(X_test_scaled)
    assert (y_pred == y_pred_loaded).all()
    print(f"   Load verified: predictions match")
    
    # 8. Experiment Tracking
    print("\n📝 8. EXPERIMENT TRACKING")
    print("-" * 40)
    tracker = ExperimentTracker(experiment_name="production_demo")
    run_id = tracker.log_run(
        params=config,
        metrics=results["test_metrics"],
        artifacts={"model": model_path},
    )
    print(f"   Run logged: {run_id}")
    results["experiment_run_id"] = run_id
    
    print("\n" + "=" * 60)
    print("PRODUCTION PIPELINE COMPLETE")
    print("=" * 60)
    print(f"   Model: {model_cfg['type']}")
    print(f"   CV Score: {results['cv_mean']:.4f}")
    print(f"   Test F1: {results['test_metrics']['f1']:.4f}")
    print(f"   Model saved: {model_path}")
    print(f"   Experiment: {run_id}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Production ML pipeline demo")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--create-config", action="store_true", help="Create sample config file")
    args = parser.parse_args()
    
    if args.create_config:
        config = create_sample_config()
        Path("config").mkdir(exist_ok=True)
        with open("config/production.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print("Created config/production.yaml")
        return
    
    if args.config:
        config = load_config(args.config)
    else:
        print("No config provided, using defaults...")
        config = create_sample_config()
    
    run_pipeline(config)
    
    print("""
🎯 EXPERT NEXT STEPS:

1. CUSTOMIZE THE CONFIG
   • Edit config/production.yaml for your use case
   • Add feature engineering steps
   • Configure different models

2. EXTEND THE PIPELINE
   • Add A/B testing (Ch17 ex02)
   • Implement custom drift detectors (Ch17 ex03)
   • Add model registry with versioning

3. AUTOMATE WITH CI/CD
   • See topics/ch16_production_software/exercises/ex07_ci_cd_integration/
   • GitHub Actions workflow with quality gates

4. DEEP DIVES
   • Property-based testing: topics/ch16_production_software/exercises/ex05_property_based_test/
   • Monitoring design: topics/ch16_production_software/exercises/ex03_monitoring_design/
   • Project structure: topics/ch16_production_software/exercises/ex04_project_reorganization/
""")


if __name__ == "__main__":
    main()
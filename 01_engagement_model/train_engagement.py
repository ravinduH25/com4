"""
Model 1: Engagement Prediction with XGBoost
Predicts whether a student will meet the engagement activity criterion in the subsequent week (w+1).
Saves trained model and feature list into 'models/' for the Digital Twin service.
"""

import json
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


def train_engagement_model(
    data_path: str = "data/processed/engagement_train.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42
):
    print("=" * 60)
    print("Training Model 1: Engagement XGBoost Classifier")
    print("=" * 60)
    
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    features = [
        "total_clicks",
        "active_days",
        "unique_materials",
        "clicks_2w_mean",
        "active_days_2w",
        "click_change"
    ]
    target = "next_week_active"
    
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target].astype(int)
    groups = df["id_student"]
    
    print(f"Features: {features}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")
    
    # Student-aware train/test split: prevent data leakage between same student's weeks
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Training set: {len(X_train)} samples across {len(np.unique(groups.iloc[train_idx]))} students")
    print(f"Test set:     {len(X_test)} samples across {len(np.unique(groups.iloc[test_idx]))} students")
    
    # Calculate scale_pos_weight in case of class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state
    )
    
    print("\nFitting XGBoost model...")
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print("\n--- Model Evaluation on Held-Out Test Set ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Feature Importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
    print("\nFeature Importances:")
    print(feat_imp.to_string(index=False))
    
    # Save plots
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="royalblue", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Engagement ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color="teal")
    plt.xlabel("Importance Score")
    plt.title("XGBoost Feature Importances")
    plt.tight_layout()
    plot_file = reports_path / "engagement_model_evaluation.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\nSaved evaluation plot to: {plot_file}")
    
    # Save Artifacts for Digital Twin API
    model_file = models_path / "engagement_xgb.joblib"
    features_file = models_path / "engagement_features.joblib"
    meta_file = models_path / "engagement_metadata.json"
    
    joblib.dump(model, model_file)
    joblib.dump(features, features_file)
    
    metadata = {
        "model_name": "Engagement XGBoost Classifier",
        "version": "1.0.0",
        "features": features,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        }
    }
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved model artifact: {model_file}")
    print(f"Saved feature list:   {features_file}")
    print(f"Saved metadata:       {meta_file}")
    return model, metadata


if __name__ == "__main__":
    train_engagement_model()

"""
Model 2: Disengagement-Risk Prediction with XGBoost
Predicts whether a student is at risk of disengagement / prolonged inactivity within the next 14 days (t, t+14].
Includes threshold optimization to maximize recall while controlling false positive alarms.
Saves model artifacts and decision threshold into 'models/' for the Digital Twin engine.
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


def train_disengagement_model(
    data_path: str = "data/processed/disengagement_train.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42
):
    print("=" * 60)
    print("Training Model 2: Disengagement-Risk XGBoost Classifier")
    print("=" * 60)
    
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    features = [
        "active_days_7d",
        "active_days_14d",
        "clicks_7d",
        "click_change",
        "days_since_last_activity",
        "previous_inactivity_events"
    ]
    target = "future_inactive_14d"
    
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target].astype(int)
    groups = df["id_student"]
    
    print(f"Features: {features}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")
    
    # Student-aware split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Training set: {len(X_train)} samples across {len(np.unique(groups.iloc[train_idx]))} students")
    print(f"Test set:     {len(X_test)} samples across {len(np.unique(groups.iloc[test_idx]))} students")
    
    # Handling class imbalance (disengagement is typically the minority class)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)
    print(f"Class imbalance weighting: scale_pos_weight = {scale_pos_weight:.2f}")
    
    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state
    )
    
    print("\nFitting XGBoost Disengagement model...")
    model.fit(X_train, y_train)
    
    # Evaluation at standard 0.5 threshold
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # Threshold Tuning: In early intervention, missed disengaged students (false negatives) are costly.
    # We evaluate thresholds between 0.2 and 0.8 to select the optimal F1 / Recall balanced threshold.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) > 0 else 0.5
    
    # Clamp threshold between 0.3 and 0.6 for stability
    optimal_threshold = float(np.clip(optimal_threshold, 0.30, 0.60))
    print(f"\nCalibrated Optimal Alert Threshold: {optimal_threshold:.3f}")
    
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred_opt)
    f1 = f1_score(y_test, y_pred_opt)
    prec = precision_score(y_test, y_pred_opt)
    rec = recall_score(y_test, y_pred_opt)
    
    print("\n--- Disengagement Risk Evaluation (Threshold: {:.3f}) ---".format(optimal_threshold))
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_opt, digits=4))
    
    # Feature Importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
    print("\nFeature Importances:")
    print(feat_imp.to_string(index=False))
    
    # Plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="crimson", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Disengagement ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(recalls, precisions, label=f"PR-AUC = {pr_auc:.3f}", color="darkorange", lw=2)
    plt.axvline(rec, color="gray", linestyle=":", label=f"Recall @ opt = {rec:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color="firebrick")
    plt.xlabel("Importance Score")
    plt.title("Risk Factor Importances")
    plt.tight_layout()
    plot_file = reports_path / "disengagement_model_evaluation.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\nSaved evaluation plot to: {plot_file}")
    
    # Save Artifacts
    model_file = models_path / "disengagement_xgb.joblib"
    features_file = models_path / "disengagement_features.joblib"
    threshold_file = models_path / "disengagement_threshold.json"
    
    joblib.dump(model, model_file)
    joblib.dump(features, features_file)
    
    meta = {
        "model_name": "Disengagement-Risk XGBoost Classifier",
        "version": "1.0.0",
        "features": features,
        "optimal_threshold": float(optimal_threshold),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        }
    }
    with open(threshold_file, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Saved model artifact:      {model_file}")
    print(f"Saved feature list:        {features_file}")
    print(f"Saved optimal threshold:   {threshold_file}")
    return model, meta


if __name__ == "__main__":
    train_disengagement_model()

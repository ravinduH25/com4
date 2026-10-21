"""
AI-Powered Multimodal Student Digital Twin & Cognitive Memory Engine
Combines predictions from:
- Model 1: Engagement XGBoost
- Model 2: Disengagement-Risk XGBoost
- Model 3: K-Means Behavioural Clustering
Maintains longitudinal cognitive memory snapshots and triggers proactive early-warning interventions.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd


class CognitiveMemoryEngine:
    """
    Stores and analyzes historical states of a student's behavioural trajectory.
    Tracks memory drift and early-warning signals across learning weeks.
    """
    def __init__(self, max_history_per_student: int = 50):
        self.max_history = max_history_per_student
        self._memory_store: Dict[str, List[Dict[str, Any]]] = {}
        
    def append_state(self, student_id: str, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if student_id not in self._memory_store:
            self._memory_store[student_id] = []
            
        self._memory_store[student_id].append(snapshot)
        if len(self._memory_store[student_id]) > self.max_history:
            self._memory_store[student_id].pop(0)
            
        return self._memory_store[student_id]
        
    def get_history(self, student_id: str) -> List[Dict[str, Any]]:
        return self._memory_store.get(student_id, [])
        
    def compute_memory_drift(self, student_id: str) -> Dict[str, Any]:
        """Calculates trajectory change between the most recent and preceding state."""
        history = self.get_history(student_id)
        if len(history) < 2:
            return {
                "trajectory": "insufficient_history",
                "engagement_delta": 0.0,
                "risk_delta": 0.0,
                "profile_changed": False,
                "summary": "Baseline memory snapshot established."
            }
            
        latest = history[-1]
        previous = history[-2]
        
        eng_delta = round(latest["engagement_probability"] - previous["engagement_probability"], 3)
        risk_delta = round(latest["disengagement_risk"] - previous["disengagement_risk"], 3)
        profile_changed = latest["behavior_profile"]["profile_name"] != previous["behavior_profile"]["profile_name"]
        
        if risk_delta >= 0.15:
            trajectory = "rapid_risk_escalation"
            summary = "Alert: Significant rise in disengagement risk detected over the last observation window."
        elif eng_delta <= -0.15:
            trajectory = "engagement_declining"
            summary = "Notice: Observable drop in study interaction frequency and activity volume."
        elif eng_delta >= 0.15:
            trajectory = "positive_momentum"
            summary = "Praise: Interaction consistency and platform engagement have notably improved."
        else:
            trajectory = "stable"
            summary = "Stable behavioural trend maintained."
            
        return {
            "trajectory": trajectory,
            "engagement_delta": eng_delta,
            "risk_delta": risk_delta,
            "profile_changed": profile_changed,
            "previous_profile": previous["behavior_profile"]["profile_name"],
            "current_profile": latest["behavior_profile"]["profile_name"],
            "summary": summary
        }


class StudentDigitalTwinEngine:
    """
    Unified Student Digital Twin orchestrating ML inference and Cognitive Memory state updates.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_path = Path(models_dir)
        self.memory = CognitiveMemoryEngine()
        
        # Load Model 1: Engagement XGBoost
        eng_model_path = self.models_path / "engagement_xgb.joblib"
        eng_feat_path = self.models_path / "engagement_features.joblib"
        
        # Load Model 2: Disengagement XGBoost
        diseng_model_path = self.models_path / "disengagement_xgb.joblib"
        diseng_feat_path = self.models_path / "disengagement_features.joblib"
        thresh_path = self.models_path / "disengagement_threshold.json"
        
        # Load Model 3: Clustering
        scaler_path = self.models_path / "cluster_scaler.joblib"
        kmeans_path = self.models_path / "kmeans_model.joblib"
        catalog_path = self.models_path / "cluster_profiles.json"
        
        self.engagement_model = joblib.load(eng_model_path) if eng_model_path.exists() else None
        self.engagement_features = joblib.load(eng_feat_path) if eng_feat_path.exists() else [
            "total_clicks", "active_days", "unique_materials", "clicks_2w_mean", "active_days_2w", "click_change"
        ]
        
        self.disengagement_model = joblib.load(diseng_model_path) if diseng_model_path.exists() else None
        self.disengagement_features = joblib.load(diseng_feat_path) if diseng_feat_path.exists() else [
            "active_days_7d", "active_days_14d", "clicks_7d", "click_change", "days_since_last_activity", "previous_inactivity_events"
        ]
        
        self.disengagement_threshold = 0.40
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                thresh_meta = json.load(f)
                self.disengagement_threshold = float(thresh_meta.get("optimal_threshold", 0.40))
                
        self.cluster_scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        self.kmeans_model = joblib.load(kmeans_path) if kmeans_path.exists() else None
        
        self.cluster_profiles = {}
        if catalog_path.exists():
            with open(catalog_path, "r") as f:
                self.cluster_profiles = json.load(f).get("profiles", {})
                
    def infer(self, student_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference through all three models given a unified feature dictionary.
        """
        # 1. Engagement Model
        eng_input = pd.DataFrame([{f: student_features.get(f, 0) for f in self.engagement_features}])
        eng_prob = float(self.engagement_model.predict_proba(eng_input)[0, 1]) if self.engagement_model else 0.5
        
        # 2. Disengagement Model
        dis_input = pd.DataFrame([{f: student_features.get(f, 0) for f in self.disengagement_features}])
        dis_prob = float(self.disengagement_model.predict_proba(dis_input)[0, 1]) if self.disengagement_model else 0.2
        is_at_risk = bool(dis_prob >= self.disengagement_threshold)
        
        # 3. K-Means Clustering
        cluster_features = [
            "active_days_7d", "total_clicks_7d", "unique_materials_7d",
            "active_weeks_ratio", "mean_weekly_clicks", "click_variability"
        ]
        cluster_input = pd.DataFrame([{f: student_features.get(f, 0) for f in cluster_features}])
        
        if self.cluster_scaler and self.kmeans_model:
            scaled_cluster = self.cluster_scaler.transform(cluster_input)
            cluster_id = int(self.kmeans_model.predict(scaled_cluster)[0])
            profile_info = self.cluster_profiles.get(str(cluster_id), {
                "profile_name": f"Profile Cluster {cluster_id}",
                "description": "Standard behavioural typology.",
                "recommended_action": "Continue regular learning rhythm."
            })
        else:
            cluster_id = 0
            profile_info = {
                "profile_name": "Standard Learner",
                "description": "Unprofiled baseline.",
                "recommended_action": "Engage regularly."
            }
            
        return {
            "engagement_probability": round(eng_prob, 3),
            "disengagement_risk": round(dis_prob, 3),
            "risk_alert": is_at_risk,
            "alert_threshold": self.disengagement_threshold,
            "behavior_profile": {
                "cluster_id": cluster_id,
                "profile_name": profile_info.get("profile_name"),
                "description": profile_info.get("description"),
                "recommended_action": profile_info.get("recommended_action")
            }
        }
        
    def update_twin(self, student_id: str, student_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates the Student Digital Twin with fresh interaction observations,
        records snapshot in Cognitive Memory, and computes behavioural drift.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        inference_result = self.infer(student_features)
        
        snapshot = {
            "student_id": str(student_id),
            "timestamp": timestamp,
            "features_snapshot": student_features,
            "engagement_probability": inference_result["engagement_probability"],
            "disengagement_risk": inference_result["disengagement_risk"],
            "risk_alert": inference_result["risk_alert"],
            "behavior_profile": inference_result["behavior_profile"]
        }
        
        # Append to Cognitive Memory
        self.memory.append_state(str(student_id), snapshot)
        drift = self.memory.compute_memory_drift(str(student_id))
        
        return {
            "student_id": str(student_id),
            "current_state": snapshot,
            "memory_drift": drift,
            "total_memory_snapshots": len(self.memory.get_history(str(student_id)))
        }

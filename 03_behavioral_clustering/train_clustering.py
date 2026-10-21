"""
Model 3: Unsupervised Behavioural Profiling with K-Means & Comparison DBSCAN
Discovers learner behavioural typologies without class labels.
Saves scaler, K-Means model, DBSCAN model, and profile definitions to 'models/'.
"""

import json
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def train_clustering_models(
    data_path: str = "data/processed/behavior_profiles.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    n_clusters_default: int = 4,
    random_state: int = 42
):
    print("=" * 60)
    print("Training Model 3: K-Means & DBSCAN Behavioural Profiling")
    print("=" * 60)
    
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(data_path)
    print(f"Loaded student profiles: {df.shape[0]} students")
    
    features = [
        "active_days_7d",
        "total_clicks_7d",
        "unique_materials_7d",
        "active_weeks_ratio",
        "mean_weekly_clicks",
        "click_variability"
    ]
    
    X = df[features].fillna(0)
    
    # 1. Standardize features
    print("\n[Step 1] Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Evaluate K-Means across k in [2, 3, 4, 5, 6]
    print("\n[Step 2] Evaluating K-Means candidate cluster counts...")
    cluster_metrics = []
    k_range = [2, 3, 4, 5, 6]
    
    best_k = n_clusters_default
    best_sil = -1.0
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        
        cluster_metrics.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "inertia": km.inertia_
        })
        print(f"  k={k}: Silhouette = {sil:.4f} (higher is better), Davies-Bouldin = {db:.4f} (lower is better), Calinski-Harabasz = {ch:.1f}")
        
        if sil > best_sil:
            best_sil = sil
            best_k = k
            
    print(f"\nSelected optimal k = {best_k} (Silhouette = {best_sil:.4f})")
    
    # Fit final K-Means with optimal or default k
    final_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    df["kmeans_cluster"] = final_kmeans.fit_predict(X_scaled)
    
    # 3. Analyze and Name Cluster Profiles
    print("\n[Step 3] Cluster Centroid Profiles (unscaled feature means):")
    centroids_df = df.groupby("kmeans_cluster")[features].mean().round(2)
    counts = df["kmeans_cluster"].value_counts().sort_index()
    centroids_df["student_count"] = counts
    print(centroids_df)
    
    # Automatically generate descriptive profile names based on relative centroid values
    profile_catalog = {}
    mean_clicks_overall = df["mean_weekly_clicks"].mean()
    mean_ratio_overall = df["active_weeks_ratio"].mean()
    
    for c_id, row in centroids_df.iterrows():
        ratio = row["active_weeks_ratio"]
        clicks = row["mean_weekly_clicks"]
        var = row["click_variability"]
        
        if ratio >= mean_ratio_overall and clicks >= mean_clicks_overall * 1.2:
            name = "Consistent High-Engager"
            desc = "Consistently high engagement across weeks with wide study material coverage."
            action = "Offer advanced extension topics and peer-mentoring leadership roles."
        elif ratio >= mean_ratio_overall and clicks < mean_clicks_overall * 1.2:
            name = "Steady Moderate Learner"
            desc = "Regular weekly participation with steady, manageable interaction volume."
            action = "Maintain regular study pace, encourage incremental challenge."
        elif var > clicks * 0.8:
            name = "Spurt / Crammer Learner"
            desc = "Sporadic high activity bursts interspersed with inactive gaps."
            action = "Provide structured pacing checkpoints and mid-week micro-tasks."
        else:
            name = "Low-Activity / At-Risk"
            desc = "Low participation frequency, limited material access, potential withdrawal risk."
            action = "Early supportive check-in, review foundational study plan."
            
        profile_catalog[int(c_id)] = {
            "profile_name": name,
            "description": desc,
            "recommended_action": action,
            "student_count": int(row["student_count"]),
            "centroid_means": row[features].to_dict()
        }
        
    print("\nCluster Profiling Catalog:")
    for cid, info in profile_catalog.items():
        print(f" Cluster {cid}: {info['profile_name']} (N={info['student_count']}) -> {info['recommended_action']}")
        
    # 4. Fit DBSCAN as comparison model
    print("\n[Step 4] Fitting Comparison Model: DBSCAN...")
    dbscan = DBSCAN(eps=1.2, min_samples=5)
    df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)
    
    db_labels = df["dbscan_cluster"]
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = int((db_labels == -1).sum())
    noise_ratio = n_noise / len(df)
    
    print(f"DBSCAN results: {n_clusters_db} dense clusters found, {n_noise} noise/outlier students ({noise_ratio:.1%})")
    
    non_noise = db_labels != -1
    if len(set(db_labels[non_noise])) >= 2 and non_noise.sum() > len(set(db_labels[non_noise])):
        db_sil = silhouette_score(X_scaled[non_noise], db_labels[non_noise])
        print(f"DBSCAN non-noise silhouette score: {db_sil:.4f}")
    else:
        db_sil = None
        print("DBSCAN produced fewer than 2 non-noise clusters for silhouette computation.")
        
    # 5. Visualizations: PCA 2D Projection
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(df["pca_x"], df["pca_y"], c=df["kmeans_cluster"], cmap="tab10", alpha=0.8, s=40)
    plt.title(f"K-Means Behavioural Profiles (k={best_k})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter1, label="Cluster ID")
    
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(df["pca_x"], df["pca_y"], c=df["dbscan_cluster"], cmap="plasma", alpha=0.8, s=40)
    plt.title(f"DBSCAN Clusters & Outliers (Noise = {n_noise})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter2, label="Cluster (-1 = Outlier)")
    
    plt.tight_layout()
    plot_file = reports_path / "behavioral_clustering_pca.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\nSaved clustering PCA visualization to: {plot_file}")
    
    # 6. Save Artifacts for Digital Twin API
    scaler_file = models_path / "cluster_scaler.joblib"
    kmeans_file = models_path / "kmeans_model.joblib"
    dbscan_file = models_path / "dbscan_model.joblib"
    catalog_file = models_path / "cluster_profiles.json"
    
    joblib.dump(scaler, scaler_file)
    joblib.dump(final_kmeans, kmeans_file)
    joblib.dump(dbscan, dbscan_file)
    
    with open(catalog_file, "w") as f:
        json.dump({
            "features": features,
            "optimal_k": best_k,
            "profiles": profile_catalog,
            "metrics": {
                "kmeans_silhouette": float(best_sil),
                "dbscan_clusters": n_clusters_db,
                "dbscan_noise_count": n_noise
            }
        }, f, indent=2)
        
    print(f"Saved scaler artifact:     {scaler_file}")
    print(f"Saved K-Means artifact:    {kmeans_file}")
    print(f"Saved DBSCAN artifact:     {dbscan_file}")
    print(f"Saved profile catalog:     {catalog_file}")
    return final_kmeans, profile_catalog


if __name__ == "__main__":
    train_clustering_models()

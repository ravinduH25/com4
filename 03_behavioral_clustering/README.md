# Model 3: Behavioural Profiling (K-Means & DBSCAN Comparison)

## 📌 Model Overview
- **Algorithms**:
  - `KMeans`: Discovers interpretable student typologies using standardized feature distributions.
  - `DBSCAN`: Density-based spatial clustering to validate cluster boundaries and isolate unusual outlier trajectories (`label = -1`).
- **Features Used**:
  - `active_days_7d`: Consistency over past week
  - `total_clicks_7d`: Volume of interactions
  - `unique_materials_7d`: Breadth of learning resources accessed
  - `active_weeks_ratio`: Long-term course regularity ($active\_weeks / total\_weeks$)
  - `mean_weekly_clicks`: Baseline interaction scale
  - `click_variability`: Standard deviation of interaction volume across weeks
- **Cluster Profiles Generated**:
  1. **Consistent High-Engager**: High regularity, broad content access, strong study momentum.
  2. **Steady Moderate Learner**: Predictable weekly cadence with balanced interaction volumes.
  3. **Spurt / Crammer Learner**: High variance with sharp deadline-driven activity spikes.
  4. **Low-Activity / At-Risk**: Infrequent participation with elevated risk of course drop-off.

---

## 🚀 How to Run in Google Colab

### Option A: Open the Notebook
Upload and open `clustering_colab_notebook.ipynb` directly in Google Colab.

### Option B: Type Code Step-by-Step in Colab Cells

#### Cell 1 — Install Dependencies
```bash
!pip install -q scikit-learn pandas numpy matplotlib seaborn joblib
```

#### Cell 2 — Clone Your GitHub Repository
```bash
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
%cd <YOUR_REPOSITORY>/TrainCode
```

#### Cell 3 — Run Data Preprocessing
```bash
!python dataset_preparation/generate_sample_data.py
!python dataset_preparation/preprocess_all.py
```

#### Cell 4 — Train Clustering & Evaluate Metrics
```bash
!python 03_behavioral_clustering/train_clustering.py
```

#### Cell 5 — View PCA Cluster Map & Outlier Detection
```python
from IPython.display import Image, display
display(Image("reports/behavioral_clustering_pca.png"))
```

#### Cell 6 — Download Trained Artifacts
```python
from google.colab import files
files.download("models/cluster_scaler.joblib")
files.download("models/kmeans_model.joblib")
files.download("models/dbscan_model.joblib")
files.download("models/cluster_profiles.json")
```

---

## 📦 Generated Output Artifacts
- `models/cluster_scaler.joblib`: StandardScaler fitted on training population
- `models/kmeans_model.joblib`: Serialized K-Means model
- `models/dbscan_model.joblib`: Fitted DBSCAN comparison model
- `models/cluster_profiles.json`: Behavioral cluster definitions and pedagogical interventions
- `reports/behavioral_clustering_pca.png`: 2D PCA projection scatter plot

# 🧠 AI-Powered Multimodal Student Digital Twin & Cognitive Memory Engine — Training Suite

Welcome to the training codebase for the **Student Digital Twin & Cognitive Memory Engine**. This repository is organized into distinct, modular folders so that you can train each model individually or run the unified training pipeline directly in **Google Colab**.

---

## 📁 Repository Directory Structure

```text
TrainCode/
├── Master_Train_All_In_Colab.ipynb         # Master Colab Notebook for 1-click training
├── requirements.txt                        # Core Python dependencies
├── build_notebooks.py                      # Notebook generator utility
│
├── dataset_preparation/                    # Data preparation & temporal feature engineering
│   ├── download_oulad.py                   # Automated OULAD dataset downloader
│   ├── generate_sample_data.py             # Instant benchmark/synthetic data generator
│   └── preprocess_all.py                   # Feature extraction for all 3 models
│
├── 01_engagement_model/                    # [Model 1] Engagement Prediction
│   ├── train_engagement.py                 # Standalone XGBoost training script
│   ├── engagement_colab_notebook.ipynb     # Dedicated Colab notebook
│   └── README.md                           # Step-by-step Colab tutorial
│
├── 02_disengagement_risk_model/            # [Model 2] Disengagement-Risk Early Warning
│   ├── train_disengagement.py              # XGBoost with threshold calibration
│   ├── disengagement_colab_notebook.ipynb  # Dedicated Colab notebook
│   └── README.md                           # Step-by-step Colab tutorial
│
├── 03_behavioral_clustering/               # [Model 3] Behavioural Profiling & Outliers
│   ├── train_clustering.py                 # K-Means & DBSCAN comparison script
│   ├── clustering_colab_notebook.ipynb     # Dedicated Colab notebook
│   └── README.md                           # Step-by-step Colab tutorial
│
└── 04_digital_twin_engine/                 # Longitudinal Digital Twin & Cognitive Memory
    ├── digital_twin_engine.py              # Production Python class for inference & memory
    ├── test_digital_twin.py                # 3-week simulation test
    └── README.md                           # Integration guide for FastAPI backend
```

---

## 📊 1. Datasets & Preprocessing

### Primary Dataset: Open University Learning Analytics Dataset (OULAD)
- **Official Source**: [Open University Analyse](https://analyse.kmi.open.ac.uk/open_dataset) or [Kaggle OULAD](https://www.kaggle.com/datasets/ankitverma2010/open-university-learning-analytics-dataset)
- **Tables Used**:
  - `studentVle.csv`: Daily student interaction clicks across VLE resources
  - `vle.csv`: Types and metadata for learning activities
  - `studentInfo.csv`: Student demographics and final course outcomes
  - `studentRegistration.csv`: Registration and withdrawal dates

### Preprocessing & Feature Engineering Logic:
Run `preprocess_all.py` to transform raw interaction clicks into structured temporal features:
1. **Model 1 Dataset (`data/processed/engagement_train.csv`)**:
   - Aggregated by student-week with rolling temporal windows.
   - Features: `total_clicks`, `active_days`, `unique_materials`, `clicks_2w_mean`, `active_days_2w`, `click_change`.
   - Target: `next_week_active` (predicting activity in week $w+1$).
2. **Model 2 Dataset (`data/processed/disengagement_train.csv`)**:
   - Checkpoints with rolling 7-day and 14-day history.
   - Features: `active_days_7d`, `active_days_14d`, `clicks_7d`, `click_change`, `days_since_last_activity`, `previous_inactivity_events`.
   - Target: `future_inactive_14d` (1 if student has zero activity in $(t, t+14]$).
3. **Model 3 Dataset (`data/processed/behavior_profiles.csv`)**:
   - Student behavioral profile vectors for unsupervised clustering: `active_days_7d`, `total_clicks_7d`, `unique_materials_7d`, `active_weeks_ratio`, `mean_weekly_clicks`, `click_variability`.

> **💡 Quickstart Note**: You don't have to wait for large downloads! Running `python dataset_preparation/generate_sample_data.py` instantly creates a realistic benchmark OULAD dataset in 2 seconds for immediate verification.

---

## 🚀 2. Step-by-Step Guide: Training in Google Colab

### Method 1: Training Models in Separate Colab Notebooks

If you prefer to train each model independently:

1. Push your `TrainCode` folder to your GitHub repository.
2. Go to [Google Colab](https://colab.research.google.com/).
3. Choose **File > Open Notebook > GitHub** (or upload the `.ipynb` file from the respective folder):
   - For Model 1: `TrainCode/01_engagement_model/engagement_colab_notebook.ipynb`
   - For Model 2: `TrainCode/02_disengagement_risk_model/disengagement_colab_notebook.ipynb`
   - For Model 3: `TrainCode/03_behavioral_clustering/clustering_colab_notebook.ipynb`

### Method 2: Typing Code Directly in a Fresh Google Colab Notebook

Open a new notebook in Colab and execute the following cells sequentially:

#### 🔹 Step 1: Clone Repository & Install Dependencies
```python
# Cell 1
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
%cd <YOUR_REPOSITORY>/TrainCode
!pip install -q -r requirements.txt
```

#### 🔹 Step 2: Download or Generate Dataset
```python
# Cell 2: If using official OULAD dataset:
# Option A: Download full OULAD from Kaggle:
# !mkdir -p data/raw
# !kaggle datasets download -d ankitverma2010/open-university-learning-analytics-dataset -p data/raw --unzip

# Option B: Run instant benchmark synthetic generator:
!python dataset_preparation/generate_sample_data.py

# Run preprocessing to construct temporal feature matrices:
!python dataset_preparation/preprocess_all.py
```

#### 🔹 Step 3: Train Model 1 (Engagement XGBoost)
```python
# Cell 3
!python 01_engagement_model/train_engagement.py

# Display evaluation plots:
from IPython.display import Image, display
display(Image("reports/engagement_model_evaluation.png"))
```

#### 🔹 Step 4: Train Model 2 (Disengagement-Risk XGBoost)
```python
# Cell 4
!python 02_disengagement_risk_model/train_disengagement.py

# Display evaluation plots:
display(Image("reports/disengagement_model_evaluation.png"))
```

#### 🔹 Step 5: Train Model 3 (K-Means & DBSCAN Clustering)
```python
# Cell 5
!python 03_behavioral_clustering/train_clustering.py

# Display PCA cluster distribution:
display(Image("reports/behavioral_clustering_pca.png"))
```

#### 🔹 Step 6: Test Unified Student Digital Twin & Cognitive Memory
```python
# Cell 6: Run longitudinal simulation test
!python 04_digital_twin_engine/test_digital_twin.py
```

#### 🔹 Step 7: Download Trained Model Artifacts
```python
# Cell 7: Zip and download all artifacts to your computer
!zip -r trained_digital_twin_models.zip models/ reports/
from google.colab import files
files.download("trained_digital_twin_models.zip")
```

---

## 🎯 3. Deploying Trained Models into NeuroLearn Backend

Once downloaded, extract the trained model files into your backend service:
```
repo/J26-SE-344-main/ai-services/digital-twin/app/models/
├── engagement_xgb.joblib
├── engagement_features.joblib
├── disengagement_xgb.joblib
├── disengagement_features.joblib
├── disengagement_threshold.json
├── cluster_scaler.joblib
├── kmeans_model.joblib
├── dbscan_model.joblib
└── cluster_profiles.json
```

Then copy `TrainCode/04_digital_twin_engine/digital_twin_engine.py` into `repo/J26-SE-344-main/ai-services/digital-twin/app/services/twin_service.py` to serve real-time predictions and cognitive memory tracking via FastAPI!

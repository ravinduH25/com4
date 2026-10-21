# Model 1: Engagement Prediction (XGBoost)

## 📌 Model Overview
- **Algorithm**: `XGBClassifier` with student-grouped cross-validation (`GroupShuffleSplit`)
- **Objective**: Estimates the probability that a student will be actively engaged in course learning during the upcoming observation window ($w+1$).
- **Features Used**:
  - `total_clicks`: Interaction volume in current week
  - `active_days`: Number of distinct active days in current week
  - `unique_materials`: Count of unique VLE resources accessed
  - `clicks_2w_mean`: Rolling 2-week average weekly clicks
  - `active_days_2w`: Rolling 2-week active days count
  - `click_change`: First difference in clicks ($clicks_t - clicks_{t-1}$)
- **Target Label**: `next_week_active` (Binary $1/0$: active days $\ge 2$ in week $t+1$)

---

## 🚀 How to Run in Google Colab

### Option A: Open the Notebook
Upload and open `engagement_colab_notebook.ipynb` directly in Google Colab.

### Option B: Type Code Step-by-Step in Colab Cells

#### Cell 1 — Install Dependencies
```bash
!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn joblib
```

#### Cell 2 — Clone Your GitHub Repository
```bash
# Clone your repository and change directory
!git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
%cd <YOUR_REPOSITORY>/TrainCode
```

#### Cell 3 — Run Data Preprocessing
```bash
# Prepare dataset (uses either real OULAD or benchmark generator)
!python dataset_preparation/generate_sample_data.py
!python dataset_preparation/preprocess_all.py
```

#### Cell 4 — Train Model
```bash
!python 01_engagement_model/train_engagement.py
```

#### Cell 5 — View Evaluation Metrics & Charts
```python
from IPython.display import Image, display
display(Image("reports/engagement_model_evaluation.png"))
```

#### Cell 6 — Download Trained Artifacts
```python
from google.colab import files
files.download("models/engagement_xgb.joblib")
files.download("models/engagement_features.joblib")
files.download("models/engagement_metadata.json")
```

---

## 📦 Generated Output Artifacts
- `models/engagement_xgb.joblib`: Serialized trained XGBoost model
- `models/engagement_features.joblib`: Exact feature ordering
- `models/engagement_metadata.json`: Model version, evaluation scores (ROC-AUC, PR-AUC, F1)
- `reports/engagement_model_evaluation.png`: ROC curve & feature importance plot

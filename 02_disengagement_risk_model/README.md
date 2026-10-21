# Model 2: Disengagement-Risk Prediction (XGBoost)

## 📌 Model Overview
- **Algorithm**: `XGBClassifier` with class imbalance compensation (`scale_pos_weight`) and decision threshold calibration.
- **Objective**: Identifies students at risk of prolonged future inactivity or withdrawal in the next 14 days $(t, t+14]$.
- **Features Used**:
  - `active_days_7d`: Active days during past 7 days
  - `active_days_14d`: Active days during past 14 days
  - `clicks_7d`: Total clicks during past 7 days
  - `click_change`: Relative drop or increase in clicks compared to prior 7 days
  - `days_since_last_activity`: Recency of student activity
  - `previous_inactivity_events`: Historical count of zero-activity windows
- **Target Label**: `future_inactive_14d` (Binary $1/0$: zero activity recorded in $(t, t+14]$)
- **Threshold Calibration**: Precision-Recall curve analysis automatically selects an optimal decision threshold (e.g. 0.40 - 0.60) to maximize recall for early support without generating excessive false alerts.

---

## 🚀 How to Run in Google Colab

### Option A: Open the Notebook
Upload and open `disengagement_colab_notebook.ipynb` directly in Google Colab.

### Option B: Type Code Step-by-Step in Colab Cells

#### Cell 1 — Install Dependencies
```bash
!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn joblib
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

#### Cell 4 — Train Model & Calibrate Early Warning Threshold
```bash
!python 02_disengagement_risk_model/train_disengagement.py
```

#### Cell 5 — View Evaluation Charts (ROC, Precision-Recall & Risk Factors)
```python
from IPython.display import Image, display
display(Image("reports/disengagement_model_evaluation.png"))
```

#### Cell 6 — Download Trained Artifacts
```python
from google.colab import files
files.download("models/disengagement_xgb.joblib")
files.download("models/disengagement_features.joblib")
files.download("models/disengagement_threshold.json")
```

---

## 📦 Generated Output Artifacts
- `models/disengagement_xgb.joblib`: Serialized risk model
- `models/disengagement_features.joblib`: Feature list
- `models/disengagement_threshold.json`: Calibrated alert threshold and evaluation metrics
- `reports/disengagement_model_evaluation.png`: ROC curve, PR curve, and risk factor rankings

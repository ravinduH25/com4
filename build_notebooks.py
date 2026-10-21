"""
Generates clean .ipynb files for each model folder and master pipeline.
"""

import json
from pathlib import Path


def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "NeuroLearn Training", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }


def main():
    # 1. Model 1 Notebook
    nb1_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Model 1: Engagement Prediction with XGBoost\n",
                "**Component**: AI-Powered Multimodal Student Digital Twin & Cognitive Memory Engine\n",
                "Predicts student engagement probability for the upcoming week based on temporal learning interaction features."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 1: Install Dependencies\n",
                "!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn joblib\n",
                "print('Dependencies installed.')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 2: Setup Workspace & Data Preprocessing\n",
                "# If cloned from GitHub, run inside TrainCode directory.\n",
                "# You can run synthetic benchmark generator or download OULAD directly:\n",
                "!python dataset_preparation/generate_sample_data.py\n",
                "!python dataset_preparation/preprocess_all.py"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 3: Train Engagement Model\n",
                "!python 01_engagement_model/train_engagement.py"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 4: View Evaluation Charts (ROC Curve & Feature Importances)\n",
                "from IPython.display import Image, display\n",
                "display(Image('reports/engagement_model_evaluation.png'))"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 5: Test Real-time Engagement Inference\n",
                "import joblib\n",
                "import pandas as pd\n",
                "\n",
                "model = joblib.load('models/engagement_xgb.joblib')\n",
                "features = joblib.load('models/engagement_features.joblib')\n",
                "\n",
                "sample_student = pd.DataFrame([{\n",
                "    'total_clicks': 55,\n",
                "    'active_days': 4,\n",
                "    'unique_materials': 6,\n",
                "    'clicks_2w_mean': 48.0,\n",
                "    'active_days_2w': 7,\n",
                "    'click_change': 12\n",
                "}])[features]\n",
                "\n",
                "prob = float(model.predict_proba(sample_student)[0, 1])\n",
                "print(f'Predicted Engagement Probability for Next Week: {prob:.1%}')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 6: Download Artifacts for Backend Service\n",
                "from google.colab import files\n",
                "files.download('models/engagement_xgb.joblib')\n",
                "files.download('models/engagement_features.joblib')\n",
                "files.download('models/engagement_metadata.json')"
            ]
        }
    ]
    with open("TrainCode/01_engagement_model/engagement_colab_notebook.ipynb", "w") as f:
        json.dump(create_nb(nb1_cells), f, indent=2)

    # 2. Model 2 Notebook
    nb2_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Model 2: Disengagement-Risk Prediction with XGBoost\n",
                "**Component**: AI-Powered Multimodal Student Digital Twin & Cognitive Memory Engine\n",
                "Predicts risk of future 14-day inactivity / withdrawal with calibrated alert thresholds."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 1: Install Dependencies\n",
                "!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn joblib\n",
                "print('Dependencies installed.')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 2: Run Data Preprocessing\n",
                "!python dataset_preparation/generate_sample_data.py\n",
                "!python dataset_preparation/preprocess_all.py"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 3: Train Disengagement Model & Tune Early Warning Threshold\n",
                "!python 02_disengagement_risk_model/train_disengagement.py"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 4: View Evaluation Charts (ROC, Precision-Recall & Risk Factors)\n",
                "from IPython.display import Image, display\n",
                "display(Image('reports/disengagement_model_evaluation.png'))"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 5: Test Risk Prediction with Calibrated Threshold\n",
                "import json\n",
                "import joblib\n",
                "import pandas as pd\n",
                "\n",
                "model = joblib.load('models/disengagement_xgb.joblib')\n",
                "features = joblib.load('models/disengagement_features.joblib')\n",
                "with open('models/disengagement_threshold.json') as f:\n",
                "    meta = json.load(f)\n",
                "threshold = meta['optimal_threshold']\n",
                "\n",
                "inactive_sample = pd.DataFrame([{\n",
                "    'active_days_7d': 0,\n",
                "    'active_days_14d': 1,\n",
                "    'clicks_7d': 0,\n",
                "    'click_change': -25,\n",
                "    'days_since_last_activity': 12,\n",
                "    'previous_inactivity_events': 1\n",
                "}])[features]\n",
                "\n",
                "risk_prob = float(model.predict_proba(inactive_sample)[0, 1])\n",
                "alert = risk_prob >= threshold\n",
                "print(f'Predicted Disengagement Risk: {risk_prob:.1%}')\n",
                "print(f'Early Warning Triggered:      {alert} (Threshold: {threshold:.2f})')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 6: Download Artifacts for Backend Service\n",
                "from google.colab import files\n",
                "files.download('models/disengagement_xgb.joblib')\n",
                "files.download('models/disengagement_features.joblib')\n",
                "files.download('models/disengagement_threshold.json')"
            ]
        }
    ]
    with open("TrainCode/02_disengagement_risk_model/disengagement_colab_notebook.ipynb", "w") as f:
        json.dump(create_nb(nb2_cells), f, indent=2)

    # 3. Model 3 Notebook
    nb3_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Model 3: Behavioural Profiling with K-Means & Comparison DBSCAN\n",
                "**Component**: AI-Powered Multimodal Student Digital Twin & Cognitive Memory Engine\n",
                "Unsupervised clustering of study interaction dynamics, density analysis, and anomaly detection."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 1: Install Dependencies\n",
                "!pip install -q scikit-learn pandas numpy matplotlib seaborn joblib\n",
                "print('Dependencies installed.')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 2: Run Data Preprocessing\n",
                "!python dataset_preparation/generate_sample_data.py\n",
                "!python dataset_preparation/preprocess_all.py"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 3: Train K-Means & DBSCAN with Metric Comparison (Silhouette / Davies-Bouldin)\n",
                "!python 03_behavioral_clustering/train_clustering.py"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 4: View PCA Cluster Projection & Outliers\n",
                "from IPython.display import Image, display\n",
                "display(Image('reports/behavioral_clustering_pca.png'))"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 5: Download Clustering Artifacts\n",
                "from google.colab import files\n",
                "files.download('models/cluster_scaler.joblib')\n",
                "files.download('models/kmeans_model.joblib')\n",
                "files.download('models/dbscan_model.joblib')\n",
                "files.download('models/cluster_profiles.json')"
            ]
        }
    ]
    with open("TrainCode/03_behavioral_clustering/clustering_colab_notebook.ipynb", "w") as f:
        json.dump(create_nb(nb3_cells), f, indent=2)

    # 4. Master Pipeline Notebook
    master_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Master Training Pipeline: Student Digital Twin & Cognitive Memory Engine\n",
                "Trains all three models (Engagement XGBoost, Disengagement XGBoost, K-Means/DBSCAN)\n",
                "and runs the end-to-end Student Digital Twin simulation in Google Colab."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 1: Environment & Package Setup\n",
                "!pip install -q xgboost scikit-learn pandas numpy matplotlib seaborn joblib\n",
                "print('Dependencies ready.')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 2: Data Preprocessing\n",
                "!python dataset_preparation/generate_sample_data.py\n",
                "!python dataset_preparation/preprocess_all.py"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 3: Train All Three Models\n",
                "!python 01_engagement_model/train_engagement.py\n",
                "!python 02_disengagement_risk_model/train_disengagement.py\n",
                "!python 03_behavioral_clustering/train_clustering.py"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 4: Run Digital Twin & Cognitive Memory Longitudinal Simulation\n",
                "!python 04_digital_twin_engine/test_digital_twin.py"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 5: Package All Models into a Zip for Download\n",
                "!zip -r trained_digital_twin_models.zip models/ reports/\n",
                "from google.colab import files\n",
                "files.download('trained_digital_twin_models.zip')"
            ]
        }
    ]
    with open("TrainCode/Master_Train_All_In_Colab.ipynb", "w") as f:
        json.dump(create_nb(master_cells), f, indent=2)

    print("All Jupyter Notebooks successfully created!")


if __name__ == "__main__":
    main()

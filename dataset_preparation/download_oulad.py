"""
Download Helper for Official Open University Learning Analytics Dataset (OULAD)
URL: https://analyse.kmi.open.ac.uk/open_dataset
Or Kaggle: https://www.kaggle.com/datasets/ankitverma2010/open-university-learning-analytics-dataset
"""

import os
import sys
import zipfile
from pathlib import Path


def download_via_kaggle(target_dir: str = "data/raw"):
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print("Checking for Kaggle API...")
    try:
        import kaggle
        print("Downloading OULAD dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            "ankitverma2010/open-university-learning-analytics-dataset",
            path=str(target_path),
            unzip=True
        )
        print(f"OULAD successfully downloaded and extracted into {target_path}")
    except Exception as e:
        print(f"Kaggle download failed or kaggle.json not configured: {e}")
        print("\nAlternative Options to obtain OULAD:")
        print("1. Direct Web Download:")
        print("   Visit: https://analyse.kmi.open.ac.uk/open_dataset")
        print("   Download the .zip file and extract CSVs into 'data/raw/'")
        print("2. Google Colab curl / gdown download:")
        print("   !mkdir -p data/raw")
        print("   !kaggle datasets download -d ankitverma2010/open-university-learning-analytics-dataset -p data/raw --unzip")
        print("3. Or run the synthetic benchmark generator:")
        print("   python dataset_preparation/generate_sample_data.py")


if __name__ == "__main__":
    download_via_kaggle()

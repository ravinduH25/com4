"""
Synthetic / Benchmark OULAD Dataset Generator
Creates realistic synthetic data matching the Open University Learning Analytics Dataset (OULAD) schema.
Use this for instant testing in Google Colab or local verification before loading the full 200MB dataset.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_oulad(output_dir: str = "data/raw", num_students: int = 300, num_weeks: int = 20, random_seed: int = 42):
    np.random.seed(random_seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic OULAD dataset ({num_students} students, {num_weeks} weeks)...")
    
    # 1. Modules & Presentations
    modules = ["AAA", "BBB", "CCC"]
    presentations = ["2013J", "2014J"]
    
    # 2. Student Info & Registration
    student_records = []
    registration_records = []
    
    genders = ["M", "F"]
    education_levels = ["Lower Than A Level", "A Level or Equivalent", "HE Qualification", "Post Graduate Qualification"]
    results = ["Pass", "Distinction", "Fail", "Withdrawn"]
    
    student_ids = [10000 + i for i in range(num_students)]
    
    student_archetypes = {}
    for sid in student_ids:
        # Archetype: 0 = High Engager, 1 = Average Regular, 2 = Crammer/Irregular, 3 = At-Risk/Dropping out
        archetype = np.random.choice([0, 1, 2, 3], p=[0.25, 0.40, 0.20, 0.15])
        student_archetypes[sid] = archetype
        
        mod = np.random.choice(modules)
        pres = np.random.choice(presentations)
        reg_date = int(np.random.normal(-20, 10))
        
        if archetype == 3:
            unreg_date = int(np.random.uniform(30, 90))
            result = "Withdrawn"
        elif archetype == 0:
            unreg_date = np.nan
            result = np.random.choice(["Pass", "Distinction"], p=[0.4, 0.6])
        elif archetype == 1:
            unreg_date = np.nan
            result = np.random.choice(["Pass", "Fail"], p=[0.75, 0.25])
        else: # Crammer
            unreg_date = np.nan
            result = np.random.choice(["Pass", "Fail"], p=[0.6, 0.4])
            
        student_records.append({
            "code_module": mod,
            "code_presentation": pres,
            "id_student": sid,
            "gender": np.random.choice(genders),
            "highest_education": np.random.choice(education_levels),
            "final_result": result
        })
        
        registration_records.append({
            "code_module": mod,
            "code_presentation": pres,
            "id_student": sid,
            "date_registration": reg_date,
            "date_unregistration": unreg_date
        })
        
    df_student_info = pd.DataFrame(student_records)
    df_registration = pd.DataFrame(registration_records)
    
    # 3. VLE Items
    vle_types = ["oucontent", "forumng", "resource", "url", "quiz", "subpage"]
    vle_records = []
    for site_id in range(1001, 1081):
        vle_records.append({
            "id_site": site_id,
            "code_module": np.random.choice(modules),
            "code_presentation": np.random.choice(presentations),
            "activity_type": np.random.choice(vle_types),
            "week_from": np.random.choice([np.nan, 1, 2, 5]),
            "week_to": np.random.choice([np.nan, 10, 20])
        })
    df_vle = pd.DataFrame(vle_records)
    site_ids = df_vle["id_site"].values
    
    # 4. Student VLE Interaction Logs
    vle_logs = []
    total_days = num_weeks * 7
    
    for _, s_row in df_student_info.iterrows():
        sid = s_row["id_student"]
        mod = s_row["code_module"]
        pres = s_row["code_presentation"]
        arch = student_archetypes[sid]
        
        for day in range(0, total_days):
            # Probability of login depends on archetype & course progression
            if arch == 0:  # High Engager: highly active throughout
                prob = 0.65
                click_mean, click_std = 25, 12
            elif arch == 1:  # Average Regular: moderate steady activity
                prob = 0.35
                click_mean, click_std = 15, 8
            elif arch == 2:  # Crammer: spikes every 2-3 weeks, low otherwise
                is_cram_week = (day // 7) in [3, 7, 11, 15, 19]
                prob = 0.70 if is_cram_week else 0.10
                click_mean, click_std = (45, 20) if is_cram_week else (6, 4)
            else:  # At-risk / Disengaged: decays to zero after day 30-50
                decay = max(0.0, 1.0 - (day / 60.0))
                prob = 0.25 * decay
                click_mean, click_std = 8, 5
                
            if np.random.rand() < prob:
                num_sites = np.random.randint(1, 6)
                chosen_sites = np.random.choice(site_ids, size=num_sites, replace=False)
                for site in chosen_sites:
                    clicks = max(1, int(np.random.normal(click_mean / num_sites, click_std / num_sites)))
                    vle_logs.append({
                        "code_module": mod,
                        "code_presentation": pres,
                        "id_student": sid,
                        "id_site": site,
                        "date": day,
                        "sum_click": clicks
                    })
                    
    df_student_vle = pd.DataFrame(vle_logs)
    
    # Save CSVs
    df_student_vle.to_csv(out_path / "studentVle.csv", index=False)
    df_student_info.to_csv(out_path / "studentInfo.csv", index=False)
    df_registration.to_csv(out_path / "studentRegistration.csv", index=False)
    df_vle.to_csv(out_path / "vle.csv", index=False)
    
    print(f"Generated raw synthetic OULAD data in '{out_path}':")
    print(f"  - studentVle.csv: {len(df_student_vle):,} interaction events")
    print(f"  - studentInfo.csv: {len(df_student_info):,} student records")
    print(f"  - studentRegistration.csv: {len(df_registration):,} registration records")
    print(f"  - vle.csv: {len(df_vle):,} learning resources")
    return out_path


if __name__ == "__main__":
    generate_synthetic_oulad()

"""
Unified Preprocessing & Temporal Feature Engineering Pipeline
Constructs feature tables for all three models:
1. data/processed/engagement_train.csv -> Model 1 (Engagement XGBoost)
2. data/processed/disengagement_train.csv -> Model 2 (Disengagement-Risk XGBoost)
3. data/processed/behavior_profiles.csv -> Model 3 (K-Means & DBSCAN Clustering)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def load_raw_tables(raw_dir: Path):
    print(f"Loading raw tables from {raw_dir}...")
    vle_path = raw_dir / "studentVle.csv"
    if not vle_path.exists():
        raise FileNotFoundError(
            f"Missing studentVle.csv in {raw_dir}. Run generate_sample_data.py or download OULAD first!"
        )
        
    student_vle = pd.read_csv(vle_path)
    
    info_path = raw_dir / "studentInfo.csv"
    reg_path = raw_dir / "studentRegistration.csv"
    vle_meta_path = raw_dir / "vle.csv"
    
    student_info = pd.read_csv(info_path) if info_path.exists() else None
    registration = pd.read_csv(reg_path) if reg_path.exists() else None
    vle_meta = pd.read_csv(vle_meta_path) if vle_meta_path.exists() else None
    
    return student_vle, student_info, registration, vle_meta


def build_engagement_dataset(student_vle: pd.DataFrame, processed_dir: Path):
    """
    Model 1: Engagement Prediction (XGBoost)
    Unit of observation: Student-Course-Week
    Target: next_week_active (whether student is active in week w+1)
    Features: total_clicks, active_days, unique_materials, clicks_2w_mean, active_days_2w, click_change
    """
    print("\n[1/3] Building Engagement training dataset...")
    df = student_vle.copy()
    
    # Filter valid dates >= 0
    df = df[df["date"] >= 0].copy()
    df["week"] = (df["date"] // 7).astype(int)
    
    keys = ["code_module", "code_presentation", "id_student"]
    
    # Aggregate to student-week
    weekly = (
        df.groupby(keys + ["week"])
        .agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            unique_materials=("id_site", "nunique")
        )
        .reset_index()
    )
    
    # Sort for temporal calculations
    weekly = weekly.sort_values(by=keys + ["week"]).reset_index(drop=True)
    
    # Calculate rolling temporal features
    records = []
    for enrolment_key, group in weekly.groupby(keys):
        g = group.sort_values("week").copy()
        
        # Build dense week grid to avoid skipping inactive weeks
        min_w, max_w = g["week"].min(), g["week"].max()
        all_weeks = pd.DataFrame({"week": range(min_w, max_w + 1)})
        merged = pd.merge(all_weeks, g, on="week", how="left")
        merged["total_clicks"] = merged["total_clicks"].fillna(0)
        merged["active_days"] = merged["active_days"].fillna(0)
        merged["unique_materials"] = merged["unique_materials"].fillna(0)
        for i, k in enumerate(keys):
            merged[k] = enrolment_key[i]
            
        merged["clicks_2w_mean"] = merged["total_clicks"].rolling(window=2, min_periods=1).mean()
        merged["active_days_2w"] = merged["active_days"].rolling(window=2, min_periods=1).sum()
        merged["click_change"] = merged["total_clicks"].diff().fillna(0)
        
        # Target: next week's activity (active_days >= 2 in week t+1)
        # Avoid data leakage: only use t+1 for the target
        merged["next_week_active"] = (merged["active_days"].shift(-1) >= 2).astype(int)
        
        # Exclude the very last week where future observation window is incomplete
        valid_rows = merged.iloc[:-1].copy()
        records.append(valid_rows)
        
    engagement_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    out_file = processed_dir / "engagement_train.csv"
    engagement_df.to_csv(out_file, index=False)
    print(f" Saved: {out_file} ({len(engagement_df):,} rows)")
    print(f" Class distribution:\n{engagement_df['next_week_active'].value_counts(normalize=True).round(3)}")
    return engagement_df


def build_disengagement_dataset(student_vle: pd.DataFrame, student_info: pd.DataFrame, processed_dir: Path):
    """
    Model 2: Disengagement-Risk Prediction (XGBoost)
    Predicts probability of an adverse event: no activity in the next 14 days (t, t+14]
    Features up to day t: active_days_7d, active_days_14d, clicks_7d, click_change, days_since_last_activity, previous_inactivity_events
    """
    print("\n[2/3] Building Disengagement-Risk training dataset...")
    df = student_vle[student_vle["date"] >= 0].copy()
    keys = ["code_module", "code_presentation", "id_student"]
    
    # Pre-aggregate daily activity per student
    daily = (
        df.groupby(keys + ["date"])
        .agg(daily_clicks=("sum_click", "sum"))
        .reset_index()
    )
    
    max_course_day = int(daily["date"].max())
    # Sampling prediction checkpoints every 7 days (e.g. Day 14, 21, 28, ... up to max_day - 14)
    checkpoints = list(range(14, max_course_day - 14, 7))
    
    rows = []
    grouped_students = daily.groupby(keys)
    
    for enrolment_key, student_daily in grouped_students:
        active_dates = set(student_daily["date"].values)
        click_dict = dict(zip(student_daily["date"], student_daily["daily_clicks"]))
        
        for t in checkpoints:
            # Past 7 days: [t-6, t]
            days_7 = [t - d for d in range(7)]
            clicks_7d = sum(click_dict.get(d, 0) for d in days_7)
            active_days_7d = sum(1 for d in days_7 if d in active_dates)
            
            # Prior 7 days: [t-13, t-7] for click change comparison
            days_prior_7 = [t - 7 - d for d in range(7)]
            clicks_prior_7d = sum(click_dict.get(d, 0) for d in days_prior_7)
            click_change = clicks_7d - clicks_prior_7d
            
            # Past 14 days: [t-13, t]
            days_14 = [t - d for d in range(14)]
            active_days_14d = sum(1 for d in days_14 if d in active_dates)
            
            # Days since last active date <= t
            past_active = [d for d in active_dates if d <= t]
            if past_active:
                days_since_last_activity = t - max(past_active)
            else:
                days_since_last_activity = t  # Never active yet
                
            # Previous zero-activity weeks prior to t
            prev_weeks_inactive = 1 if active_days_14d == 0 else 0
            
            # Target: Future 14 days (t, t+14] -> 1 if zero active days, else 0
            future_14_days = [t + 1 + d for d in range(14)]
            future_active_count = sum(1 for d in future_14_days if d in active_dates)
            future_inactive_14d = 1 if future_active_count == 0 else 0
            
            rows.append({
                "code_module": enrolment_key[0],
                "code_presentation": enrolment_key[1],
                "id_student": enrolment_key[2],
                "t_day": t,
                "active_days_7d": active_days_7d,
                "active_days_14d": active_days_14d,
                "clicks_7d": clicks_7d,
                "click_change": click_change,
                "days_since_last_activity": days_since_last_activity,
                "previous_inactivity_events": prev_weeks_inactive,
                "future_inactive_14d": future_inactive_14d
            })
            
    disengagement_df = pd.DataFrame(rows)
    out_file = processed_dir / "disengagement_train.csv"
    disengagement_df.to_csv(out_file, index=False)
    print(f" Saved: {out_file} ({len(disengagement_df):,} rows)")
    print(f" Class distribution (1 = future inactive 14d):\n{disengagement_df['future_inactive_14d'].value_counts(normalize=True).round(3)}")
    return disengagement_df


def build_clustering_dataset(student_vle: pd.DataFrame, processed_dir: Path):
    """
    Model 3: Unsupervised Behavioural Profiling (K-Means & DBSCAN)
    Features: active_days_7d, total_clicks_7d, unique_materials_7d, active_weeks_ratio, mean_weekly_clicks, click_variability
    """
    print("\n[3/3] Building Behavioural Clustering dataset...")
    df = student_vle[student_vle["date"] >= 0].copy()
    df["week"] = (df["date"] // 7).astype(int)
    keys = ["code_module", "code_presentation", "id_student"]
    
    total_course_weeks = max(1, int(df["week"].max()))
    
    weekly_agg = (
        df.groupby(keys + ["week"])
        .agg(
            week_clicks=("sum_click", "sum"),
            week_active_days=("date", "nunique"),
            week_materials=("id_site", "nunique")
        )
        .reset_index()
    )
    
    cluster_records = []
    # Most recent week in the log for 7-day features
    max_week = weekly_agg["week"].max()
    
    for enrolment_key, g in weekly_agg.groupby(keys):
        active_weeks_count = len(g)
        active_weeks_ratio = active_weeks_count / float(total_course_weeks)
        mean_weekly_clicks = g["week_clicks"].mean()
        click_variability = g["week_clicks"].std() if len(g) > 1 else 0.0
        
        # Recent week slice (last observable week for this student)
        recent_record = g[g["week"] == g["week"].max()].iloc[0]
        
        cluster_records.append({
            "code_module": enrolment_key[0],
            "code_presentation": enrolment_key[1],
            "id_student": enrolment_key[2],
            "active_days_7d": recent_record["week_active_days"],
            "total_clicks_7d": recent_record["week_clicks"],
            "unique_materials_7d": recent_record["week_materials"],
            "active_weeks_ratio": round(active_weeks_ratio, 4),
            "mean_weekly_clicks": round(mean_weekly_clicks, 2),
            "click_variability": round(click_variability, 2)
        })
        
    profiles_df = pd.DataFrame(cluster_records).fillna(0)
    out_file = processed_dir / "behavior_profiles.csv"
    profiles_df.to_csv(out_file, index=False)
    print(f" Saved: {out_file} ({len(profiles_df):,} student profiles)")
    print(profiles_df.head())
    return profiles_df


def run_pipeline(raw_path: str = "data/raw", processed_path: str = "data/processed"):
    raw_dir = Path(raw_path)
    processed_dir = Path(processed_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    student_vle, student_info, registration, vle_meta = load_raw_tables(raw_dir)
    build_engagement_dataset(student_vle, processed_dir)
    build_disengagement_dataset(student_vle, student_info, processed_dir)
    build_clustering_dataset(student_vle, processed_dir)
    print("\n Preprocessing successfully completed for all models!")


if __name__ == "__main__":
    run_pipeline()

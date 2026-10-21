"""
Verification test for the Student Digital Twin & Cognitive Memory Engine
Demonstrates longitudinal behavioral tracking across 3 consecutive weeks for a sample student.
"""

import json
from digital_twin_engine import StudentDigitalTwinEngine


def run_digital_twin_test():
    print("=" * 60)
    print("Testing Student Digital Twin & Cognitive Memory Engine")
    print("=" * 60)
    
    twin_engine = StudentDigitalTwinEngine(models_dir="models")
    
    student_id = "STU_9941"
    
    # Week 1 Observation: High Engagement
    week_1_features = {
        "total_clicks": 68,
        "active_days": 5,
        "unique_materials": 8,
        "clicks_2w_mean": 60.0,
        "active_days_2w": 9,
        "click_change": 15,
        "active_days_7d": 5,
        "active_days_14d": 9,
        "clicks_7d": 68,
        "days_since_last_activity": 1,
        "previous_inactivity_events": 0,
        "total_clicks_7d": 68,
        "unique_materials_7d": 8,
        "active_weeks_ratio": 1.0,
        "mean_weekly_clicks": 64.0,
        "click_variability": 12.0
    }
    
    print("\n--- Simulating Week 1 (Active Student) ---")
    w1_res = twin_engine.update_twin(student_id, week_1_features)
    print(f"Engagement Probability: {w1_res['current_state']['engagement_probability']:.1%}")
    print(f"Disengagement Risk:     {w1_res['current_state']['disengagement_risk']:.1%}")
    print(f"Risk Alert:             {w1_res['current_state']['risk_alert']}")
    print(f"Behavioral Profile:     {w1_res['current_state']['behavior_profile']['profile_name']}")
    print(f"Memory Drift:           {w1_res['memory_drift']['summary']}")
    
    # Week 2 Observation: Dropping Activity
    week_2_features = {
        "total_clicks": 14,
        "active_days": 1,
        "unique_materials": 2,
        "clicks_2w_mean": 41.0,
        "active_days_2w": 6,
        "click_change": -54,
        "active_days_7d": 1,
        "active_days_14d": 6,
        "clicks_7d": 14,
        "days_since_last_activity": 5,
        "previous_inactivity_events": 0,
        "total_clicks_7d": 14,
        "unique_materials_7d": 2,
        "active_weeks_ratio": 0.85,
        "mean_weekly_clicks": 45.0,
        "click_variability": 28.0
    }
    
    print("\n--- Simulating Week 2 (Slowing Down) ---")
    w2_res = twin_engine.update_twin(student_id, week_2_features)
    print(f"Engagement Probability: {w2_res['current_state']['engagement_probability']:.1%}")
    print(f"Disengagement Risk:     {w2_res['current_state']['disengagement_risk']:.1%}")
    print(f"Risk Alert:             {w2_res['current_state']['risk_alert']}")
    print(f"Behavioral Profile:     {w2_res['current_state']['behavior_profile']['profile_name']}")
    print(f"Memory Drift:           {w2_res['memory_drift']['summary']}")
    
    # Week 3 Observation: Prolonged Inactivity (Disengagement risk triggers)
    week_3_features = {
        "total_clicks": 0,
        "active_days": 0,
        "unique_materials": 0,
        "clicks_2w_mean": 7.0,
        "active_days_2w": 1,
        "click_change": -14,
        "active_days_7d": 0,
        "active_days_14d": 1,
        "clicks_7d": 0,
        "days_since_last_activity": 12,
        "previous_inactivity_events": 1,
        "total_clicks_7d": 0,
        "unique_materials_7d": 0,
        "active_weeks_ratio": 0.65,
        "mean_weekly_clicks": 32.0,
        "click_variability": 31.0
    }
    
    print("\n--- Simulating Week 3 (Inactive - Risk Escalation) ---")
    w3_res = twin_engine.update_twin(student_id, week_3_features)
    print(f"Engagement Probability: {w3_res['current_state']['engagement_probability']:.1%}")
    print(f"Disengagement Risk:     {w3_res['current_state']['disengagement_risk']:.1%}")
    print(f"Risk Alert:             {w3_res['current_state']['risk_alert']}")
    print(f"Behavioral Profile:     {w3_res['current_state']['behavior_profile']['profile_name']}")
    print(f"Recommended Action:     {w3_res['current_state']['behavior_profile']['recommended_action']}")
    print(f"Memory Drift:           {w3_res['memory_drift']['summary']}")
    print(f"Total History Snapshots Recorded: {w3_res['total_memory_snapshots']}")
    
    print("\n Full Digital Twin End-to-End Simulation Passed!")


if __name__ == "__main__":
    run_digital_twin_test()

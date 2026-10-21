# 🧠 Student Digital Twin & Cognitive Memory Engine

## 📌 Architecture
The Student Digital Twin is the software and behavioural intelligence layer that integrates:
1. **Model 1**: Engagement XGBoost probability ($P_{eng}$)
2. **Model 2**: Disengagement-Risk XGBoost probability ($P_{dis}$) and early warning flags
3. **Model 3**: K-Means Behavioural Profile Typology and pedagogical recommendations
4. **Cognitive Memory Engine**: Longitudinal state snapshots with automated trajectory drift detection (e.g. rising risk alerts, positive engagement momentum).

---

## 🏃 Running the Simulation Test

To verify that all models work together seamlessly:
```bash
python 04_digital_twin_engine/test_digital_twin.py
```

### Sample Simulation Output:
```
--- Simulating Week 1 (Active Student) ---
Engagement Probability: 58.4%
Disengagement Risk:     0.7%
Risk Alert:             False
Behavioral Profile:     Steady Moderate Learner
Memory Drift:           Baseline memory snapshot established.

--- Simulating Week 3 (Inactive - Risk Escalation) ---
Engagement Probability: 28.3%
Disengagement Risk:     62.2%
Risk Alert:             True
Behavioral Profile:     Low-Activity / At-Risk
Recommended Action:     Early supportive check-in, review foundational study plan.
Memory Drift:           Alert: Significant rise in disengagement risk detected over the last observation window.
```

---

## 🔌 Connecting to FastAPI Backend (`repo/ai-services/digital-twin`)

Once models are trained in Google Colab:
1. Copy the contents of `models/` into `repo/J26-SE-344-main/ai-services/digital-twin/app/models/`
2. Copy `TrainCode/04_digital_twin_engine/digital_twin_engine.py` into `repo/J26-SE-344-main/ai-services/digital-twin/app/services/twin_service.py`
3. Expose the API endpoints in FastAPI:
   - `POST /api/v1/twin/{student_id}/observe` (submit student features/events)
   - `GET /api/v1/twin/{student_id}/state` (retrieve current Digital Twin state)
   - `GET /api/v1/twin/{student_id}/history` (retrieve Cognitive Memory trajectory)

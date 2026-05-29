"""
generate_data.py — Synthetic student dataset generator
Produces a realistic CSV used for training all ML models.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

def generate_dataset():
    study_hours   = np.clip(np.random.normal(5, 2.5, N), 0, 14)
    attendance    = np.clip(np.random.normal(75, 15, N), 30, 100)
    prev_scores   = np.clip(np.random.normal(65, 15, N), 20, 100)
    assignments   = np.clip(np.random.normal(70, 18, N), 0, 100)
    sleep_hours   = np.clip(np.random.normal(7, 1.5, N), 3, 12)
    parental_edu  = np.random.choice([0, 1, 2, 3], N, p=[0.15, 0.30, 0.35, 0.20])  # none/hs/college/grad
    extra_act     = np.random.choice([0, 1], N, p=[0.45, 0.55])
    internet      = np.random.choice([0, 1], N, p=[0.20, 0.80])
    tutoring      = np.random.choice([0, 1], N, p=[0.65, 0.35])
    motivation    = np.random.randint(1, 11, N).astype(float)

    # Composite score formula (deterministic signal + noise)
    score = (
        0.30 * prev_scores +
        0.20 * attendance +
        0.18 * (study_hours / 14 * 100) +
        0.12 * assignments +
        0.06 * (sleep_hours / 12 * 100) +
        0.05 * parental_edu * 20 +
        0.04 * extra_act * 10 +
        0.03 * internet * 8 +
        0.02 * tutoring * 12 +
        0.03 * (motivation / 10 * 20) +
        np.random.normal(0, 4, N)          # noise
    )
    score = np.clip(score, 0, 100)

    # Grade labels
    def grade(s):
        if s >= 90: return "A"
        elif s >= 75: return "B"
        elif s >= 60: return "C"
        elif s >= 45: return "D"
        else: return "F"

    df = pd.DataFrame({
        "study_hours_per_day": study_hours.round(1),
        "attendance_rate":     attendance.round(1),
        "previous_score":      prev_scores.round(1),
        "assignments_score":   assignments.round(1),
        "sleep_hours":         sleep_hours.round(1),
        "parental_education":  parental_edu,          # 0-3
        "extracurricular":     extra_act,              # 0/1
        "internet_access":     internet,               # 0/1
        "tutoring":            tutoring,               # 0/1
        "motivation_level":    motivation.astype(int),
        "final_score":         score.round(2),
        "grade":               [grade(s) for s in score],
    })
    return df

if __name__ == "__main__":
    df = generate_dataset()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/student_data.csv", index=False)
    print(f"Generated {len(df)} records → data/student_data.csv")
    print(df.describe())

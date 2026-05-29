"""
app.py — Student Performance Predictor
Flask backend serving the dashboard, prediction API, and analytics.
"""
import os, json
import threading
import time
import webbrowser
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load artefacts ──────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

SCALER  = joblib.load(os.path.join(BASE, "models/scaler.pkl"))
LE      = joblib.load(os.path.join(BASE, "models/label_encoder.pkl"))
GRADE_CLF = joblib.load(os.path.join(BASE, "models/grade_classifier.pkl"))

with open(os.path.join(BASE, "models/model_results.json")) as f:
    MODEL_RESULTS = json.load(f)
with open(os.path.join(BASE, "models/feature_importance.json")) as f:
    FEAT_IMP = json.load(f)
with open(os.path.join(BASE, "models/features.json")) as f:
    FEATURES = json.load(f)

REG_MODELS = {}
for name in ["linear_regression", "ridge_regression", "random_forest",
             "gradient_boosting", "svm", "knn", "decision_tree"]:
    REG_MODELS[name] = joblib.load(os.path.join(BASE, f"models/{name}.pkl"))

DF = pd.read_csv(os.path.join(BASE, "data/student_data.csv"))


def grade_label(score):
    if score >= 90: return "A", "Excellent"
    elif score >= 75: return "B", "Good"
    elif score >= 60: return "C", "Average"
    elif score >= 45: return "D", "Below Average"
    else: return "F", "Needs Improvement"


def recommendations(data: dict, score: float) -> list:
    recs = []
    if data["study_hours_per_day"] < 4:
        recs.append("📚 Increase daily study hours to at least 4–6 hours for better retention.")
    if data["attendance_rate"] < 75:
        recs.append("🏫 Improve class attendance — target ≥ 80% to stay on track.")
    if data["sleep_hours"] < 6 or data["sleep_hours"] > 9:
        recs.append("😴 Aim for 7–9 hours of sleep per night to boost cognitive performance.")
    if data["assignments_score"] < 60:
        recs.append("📝 Focus on completing assignments — they directly impact your grade.")
    if data["motivation_level"] < 5:
        recs.append("🎯 Try goal-setting or study-group techniques to stay motivated.")
    if data["tutoring"] == 0 and score < 60:
        recs.append("🙋 Consider enrolling in tutoring sessions for personalised support.")
    if not recs:
        recs.append("✅ Keep up the excellent work! Maintain consistency in all areas.")
    return recs


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    model_name = body.get("model", "random_forest")

    row = {f: float(body.get(f, 0)) for f in FEATURES}
    X = pd.DataFrame([row])[FEATURES]

    needs_scale = model_name in ("svm", "knn", "linear_regression", "ridge_regression")
    Xs = SCALER.transform(X) if needs_scale else X.values

    model = REG_MODELS.get(model_name, REG_MODELS["random_forest"])
    score = float(np.clip(model.predict(Xs)[0], 0, 100))

    grade, label = grade_label(score)
    recs = recommendations(row, score)

    # classifier grade
    Xsc = SCALER.transform(X)
    clf_grade_idx = GRADE_CLF.predict(Xsc)[0]
    clf_grade = LE.inverse_transform([clf_grade_idx])[0]

    return jsonify({
        "predicted_score": round(score, 2),
        "grade": grade,
        "grade_label": label,
        "classifier_grade": clf_grade,
        "recommendations": recs,
        "model_used": model_name,
        "model_r2": MODEL_RESULTS.get(model_name, {}).get("r2", "N/A"),
    })


@app.route("/api/analytics")
def analytics():
    grade_dist = DF["grade"].value_counts().to_dict()
    score_hist = np.histogram(DF["final_score"], bins=10)
    corr = DF[FEATURES + ["final_score"]].corr()["final_score"].drop("final_score")
    corr_dict = {k: round(float(v), 4) for k, v in corr.items()}

    avg_by_attendance = (
        DF.assign(att_band=pd.cut(DF["attendance_rate"],
                                  bins=[0,50,60,70,80,90,100],
                                  labels=["<50","50-60","60-70","70-80","80-90","90+"]))
        .groupby("att_band", observed=True)["final_score"].mean().round(2).to_dict()
    )
    avg_by_study = (
        DF.assign(sh_band=pd.cut(DF["study_hours_per_day"],
                                 bins=[0,2,4,6,8,14],
                                 labels=["0-2","2-4","4-6","6-8","8+"]))
        .groupby("sh_band", observed=True)["final_score"].mean().round(2).to_dict()
    )

    return jsonify({
        "grade_distribution": grade_dist,
        "score_histogram": {
            "bins": [round(float(x), 1) for x in score_hist[1].tolist()],
            "counts": score_hist[0].tolist(),
        },
        "feature_correlations": corr_dict,
        "feature_importance": FEAT_IMP,
        "model_comparison": MODEL_RESULTS,
        "avg_score_by_attendance": avg_by_attendance,
        "avg_score_by_study_hours": avg_by_study,
        "dataset_stats": {
            "total_students": len(DF),
            "mean_score": round(float(DF["final_score"].mean()), 2),
            "std_score":  round(float(DF["final_score"].std()),  2),
            "max_score":  round(float(DF["final_score"].max()),  2),
            "min_score":  round(float(DF["final_score"].min()),  2),
        },
    })


@app.route("/api/compare", methods=["POST"])
def compare():
    """Predict with ALL models and return comparison."""
    body = request.get_json(force=True)
    row = {f: float(body.get(f, 0)) for f in FEATURES}
    X = pd.DataFrame([row])[FEATURES]
    Xs = SCALER.transform(X)

    out = {}
    for name, mdl in REG_MODELS.items():
        needs_scale = name in ("svm", "knn", "linear_regression", "ridge_regression")
        pred = float(np.clip(mdl.predict(Xs if needs_scale else X.values)[0], 0, 100))
        g, lbl = grade_label(pred)
        out[name] = {"score": round(pred, 2), "grade": g,
                     "r2": MODEL_RESULTS.get(name, {}).get("r2", "N/A")}
    return jsonify(out)


def open_browser(url="http://127.0.0.1:5000"):
    print(f"Attempting to open browser at {url}...")
    try:
        if os.name == "nt":
            os.startfile(url)
            return True
    except Exception as exc:
        print(f"Windows os.startfile() failed: {exc}")
    try:
        if webbrowser.open_new_tab(url):
            return True
        if webbrowser.open(url):
            return True
    except Exception as exc:
        print(f"Could not open browser automatically. Visit {url} manually. Error: {exc}")
    print(f"Browser auto-open failed; open {url} manually.")
    return False


def launch_browser(delay: float = 2.0):
    def _open():
        time.sleep(delay)
        open_browser()
    timer = threading.Timer(delay, _open)
    timer.daemon = True
    timer.start()


if os.environ.get("FLASK_RUN_FROM_CLI") and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    launch_browser()


if __name__ == "__main__":
    print("Starting Student Performance Predictor on http://127.0.0.1:5000")
    launch_browser()
    app.run(debug=True, port=5000, use_reloader=False)

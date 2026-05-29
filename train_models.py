"""
train_models.py — Trains & serialises multiple ML models
Models: Linear Regression, Random Forest, Gradient Boosting,
        SVM (regression), Decision Tree, KNN
Also produces a grade classifier (Random Forest).
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, classification_report)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

FEATURES = [
    "study_hours_per_day", "attendance_rate", "previous_score",
    "assignments_score", "sleep_hours", "parental_education",
    "extracurricular", "internet_access", "tutoring", "motivation_level",
]
TARGET_REG   = "final_score"
TARGET_CLASS = "grade"

def load_data():
    df = pd.read_csv("data/student_data.csv")
    X  = df[FEATURES]
    y_reg   = df[TARGET_REG]
    le = LabelEncoder()
    y_cls = le.fit_transform(df[TARGET_CLASS])
    return X, y_reg, y_cls, le, df

def train():
    os.makedirs("models", exist_ok=True)
    X, y_reg, y_cls, le, df = load_data()

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_cls, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── Regression models ────────────────────────────────────────────────
    reg_models = {
        "linear_regression":    LinearRegression(),
        "ridge_regression":     Ridge(alpha=1.0),
        "decision_tree":        DecisionTreeRegressor(max_depth=8, random_state=42),
        "random_forest":        RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "gradient_boosting":    GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
        "svm":                  SVR(kernel="rbf", C=10, epsilon=0.2),
        "knn":                  KNeighborsRegressor(n_neighbors=7),
    }

    results = {}
    for name, mdl in reg_models.items():
        needs_scale = name in ("svm", "knn", "linear_regression", "ridge_regression")
        Xtr = X_tr_s if needs_scale else X_tr.values
        Xte = X_te_s if needs_scale else X_te.values

        mdl.fit(Xtr, yr_tr)
        preds = mdl.predict(Xte)
        rmse  = np.sqrt(mean_squared_error(yr_te, preds))
        r2    = r2_score(yr_te, preds)
        cv    = cross_val_score(mdl, Xtr, yr_tr, cv=5, scoring="r2").mean()

        results[name] = {"rmse": round(rmse, 3), "r2": round(r2, 4), "cv_r2": round(cv, 4)}
        joblib.dump(mdl, f"models/{name}.pkl")
        print(f"{name:25s}  RMSE={rmse:.3f}  R²={r2:.4f}  CV-R²={cv:.4f}")

    # ── Grade classifier ─────────────────────────────────────────────────
    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_tr_s, yc_tr)
    yc_pred = clf.predict(X_te_s)
    clf_acc  = accuracy_score(yc_te, yc_pred)
    results["grade_classifier"] = {"accuracy": round(clf_acc, 4)}
    joblib.dump(clf, "models/grade_classifier.pkl")
    print(f"{'grade_classifier':25s}  Acc={clf_acc:.4f}")
    present = sorted(set(yc_te) | set(yc_pred))
    print(classification_report(yc_te, yc_pred, labels=present,
                                target_names=le.classes_[present]))

    # ── Feature importance (RF) ───────────────────────────────────────────
    rf = joblib.load("models/random_forest.pkl")
    importance = dict(zip(FEATURES, rf.feature_importances_.tolist()))

    # ── Persist artefacts ─────────────────────────────────────────────────
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le,     "models/label_encoder.pkl")
    with open("models/model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("models/feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    with open("models/features.json", "w") as f:
        json.dump(FEATURES, f)

    print("\n✅ All models saved to models/")
    return results, importance

if __name__ == "__main__":
    train()

# 🎓 Student Performance Predictor

> A machine learning web application that predicts student academic performance, identifies key influencing factors, and delivers personalised improvement recommendations.

---

## 📌 Objective

Educational institutions struggle to proactively identify at-risk students using only past exam data. This project builds a **data-driven prediction system** that uses academic and behavioural features to forecast student final scores and grades, enabling timely, targeted interventions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 Score Prediction | Predict final scores using 7 ML algorithms |
| 🏷️ Grade Classification | Dedicated Random Forest grade classifier (A–F) |
| 📊 Interactive Dashboard | Dataset stats, grade & score distribution charts |
| 🔄 Model Comparison | Side-by-side prediction and R² comparison across all models |
| 📈 Advanced Analytics | Correlation heatmap, feature importance, model benchmarks |
| 📋 Recommendations | Personalised improvement tips based on input profile |
| 🎛️ Interactive Form | Real-time range sliders and dropdowns for student inputs |

---

## 🤖 ML Models Included

| Model | Type | Notes |
|---|---|---|
| Linear Regression | Regression | Baseline |
| Ridge Regression | Regression | L2-regularised linear |
| Decision Tree | Regression | Interpretable |
| Random Forest | Regression | Best overall performance |
| Gradient Boosting | Regression | High accuracy |
| SVM (RBF kernel) | Regression | Robust to outliers |
| KNN | Regression | Instance-based |
| Random Forest | **Classifier** | Grade (A/B/C/D/F) |

---

## 🗂️ Tech Stack

**Backend**
- Python 3.12
- Flask 3.x — REST API & templating
- scikit-learn — model training, cross-validation
- pandas / numpy — data processing
- joblib — model serialisation

**Frontend**
- Vanilla HTML5 / CSS3 / JavaScript (ES2023)
- Chart.js 4 — interactive charts
- Responsive, dark-mode UI (no framework dependency)

---

## 📁 Project Structure

```
student-performance-predictor/
├── app.py                  # Flask application & API routes
├── generate_data.py        # Synthetic dataset generator (1 000 records)
├── train_models.py         # Model training & serialisation pipeline
├── requirements.txt
├── README.md
├── data/
│   └── student_data.csv    # Generated dataset
├── models/
│   ├── *.pkl               # Serialised models, scaler, encoder
│   ├── model_results.json  # Test-set metrics
│   ├── feature_importance.json
│   └── features.json
└── templates/
    └── index.html          # Full-featured SPA dashboard
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate dataset & train models
```bash
python generate_data.py   # Creates data/student_data.csv
python train_models.py    # Trains all models → models/*.pkl
```

### 5. Run the application
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## 📊 Input Features

| Feature | Range / Type | Description |
|---|---|---|
| Study Hours/Day | 0 – 14 hrs | Daily self-study time |
| Attendance Rate | 30 – 100 % | Class attendance percentage |
| Previous Score | 0 – 100 | Last exam or semester score |
| Assignments Score | 0 – 100 | Average assignment marks |
| Sleep Hours | 3 – 12 hrs | Nightly sleep duration |
| Motivation Level | 1 – 10 | Self-reported motivation |
| Parental Education | 0–3 (None→Grad) | Highest parental qualification |
| Extracurricular | 0 / 1 | Participates in activities |
| Internet Access | 0 / 1 | Has home internet |
| Tutoring | 0 / 1 | Enrolled in tutoring |

---

## 📡 API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Main dashboard UI |
| POST | `/api/predict` | Predict with a selected model |
| POST | `/api/compare` | Predict with all 7 models |
| GET | `/api/analytics` | Dataset statistics & chart data |

### Example — `/api/predict`
```json
POST /api/predict
{
  "study_hours_per_day": 6,
  "attendance_rate": 85,
  "previous_score": 72,
  "assignments_score": 78,
  "sleep_hours": 7.5,
  "motivation_level": 7,
  "parental_education": 2,
  "extracurricular": 1,
  "internet_access": 1,
  "tutoring": 0,
  "model": "random_forest"
}
```
```json
{
  "predicted_score": 64.3,
  "grade": "C",
  "grade_label": "Average",
  "classifier_grade": "C",
  "recommendations": ["..."],
  "model_used": "random_forest",
  "model_r2": 0.658
}
```

---

## 📸 Screenshots

<img width="1188" height="880" alt="Screenshot 2026-05-29 101145" src="https://github.com/user-attachments/assets/50b1c3a5-37b4-4d71-a780-352a2b91441d" />


---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — ML algorithms
- [Chart.js](https://www.chartjs.org/) — Data visualisation
- [Flask](https://flask.palletsprojects.com/) — Web framework

---

## 📄 License

MIT License — free to use and modify.

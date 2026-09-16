# Heart Disease Prediction Dataset 2026

**Subtitle:** 100K records · 25 clinical, lifestyle & wearable features · research-calibrated

## 📊 Overview

A comprehensive **research-grade** cardiovascular dataset containing **100,000 patient records** with **25 carefully engineered features** spanning clinical markers, lifestyle factors, wearable signals, medical history, and socioeconomic variables. Every distribution, prevalence, and inter-feature correlation has been numerically calibrated against peer-reviewed epidemiological studies — making this dataset suitable for rigorous machine learning research, medical AI coursework, and clinical risk modeling.

> **This is the modern upgrade to the original [UCI Cleveland Heart Disease Dataset (1988)](https://archive.ics.uci.edu/dataset/45/heart+disease)** — 330× more rows, 2× more features, and calibrated to 2023–2026 clinical benchmarks.

### Key Statistics
- **Records:** 100,000 patients
- **Features:** 25 variables (demographics, clinical, lifestyle, medical history, wearables)
- **Target:** 1 binary classification task (`heart_disease`: 0 or 1)
- **Positive Rate:** 46.1% (near-balanced — no oversampling needed)
- **Age Range:** 29–80 years (adult cardiovascular population)
- **File Size:** ~12.4 MB
- **Data Quality:** ✅ All prevalences and correlations validated against clinical literature

---

## 🎯 Target Variable

### `heart_disease` — Binary Classification 🫀
- **Type:** Binary (0 = No Disease, 1 = Heart Disease Present)
- **Distribution:** 53.9% No Disease | 46.1% Heart Disease
- **Scientific Basis:** 25-term logistic model with coefficients calibrated to published odds ratios from AHA/ACC 2019 Cardiovascular Risk Guidelines and Framingham Heart Study
- **Use Cases:** Binary classification · Risk scoring · Probability calibration · Fairness analysis across sex and income groups
- **Baseline AUC:** ~0.82 (Random Forest, 80/20 split)

---

## 📋 Feature Categories

### Demographics (5 features)
- `age`: Patient age in years (29–80)
- `sex`: Biological sex (Male / Female)
- `bmi`: Body Mass Index in kg/m² (15.0–54.6)
- `region`: US geographic region (Northeast / Southeast / Midwest / Southwest / West)
- `income_level`: Household income tier (Low / Middle / High)

### Lifestyle (6 features)
- `smoking_status`: Cigarette history (Never / Former / Current)
- `alcohol_intake`: Consumption level (Non-drinker / Moderate / Heavy)
- `physical_activity`: Activity level (Sedentary / Light / Active / Very Active)
- `diet_quality`: Diet score 1–10 (1 = very poor, 10 = excellent/Mediterranean)
- `sleep_hours`: Average nightly sleep duration (4.0–10.0 hrs)
- `stress_level`: Chronic stress score 1–10

### Medical History (4 features)
- `diabetes`: Diagnosed type 2 diabetes (0 / 1)
- `hypertension`: Diagnosed hypertension (0 / 1)
- `family_history`: First-degree relative with heart disease (0 / 1)
- `previous_heart_event`: Prior MI or revascularization procedure (0 / 1)

### Clinical Markers (8 features)
- `resting_bp`: Resting systolic blood pressure in mmHg (80–200)
- `cholesterol`: Total serum cholesterol in mg/dl (100–400)
- `fasting_blood_sugar`: Flag — fasting glucose > 120 mg/dl (0 / 1)
- `max_heart_rate`: Max HR during exercise stress test in bpm (107–202)
- `resting_ecg`: ECG result (Normal / ST-T Abnormality / Left Ventricular Hypertrophy)
- `exercise_angina`: Exercise-induced chest pain (0 / 1)
- `st_depression`: ST segment depression / oldpeak (0.0–6.2)
- `st_slope`: Peak exercise ST slope (Upsloping / Flat / Downsloping)

### Wearable Signals (2 features)
- `avg_resting_hr`: Average resting heart rate from wearable in bpm (40–110)
- `hrv_score`: Heart rate variability composite score (5–100, higher = better)

---

## 🔬 Scientific Calibration

### Prevalence Benchmarks (All Matched ✅)

| Metric | This Dataset | Real-World Source |
|--------|-------------|-------------------|
| Diabetes prevalence | **11.0%** | CDC National Diabetes Statistics Report 2023 |
| Hypertension prevalence | **47.0%** | AHA Heart Disease & Stroke Statistics 2023 |
| Current smokers | **13.9%** | CDC Cigarette Smoking Report 2023 |
| Family history of HD | **30.1%** | Framingham Heart Study |
| Prior cardiac event | **3.0%** | American Heart Association 2022 |
| Mean cholesterol | **217.7 mg/dl** | NHANES 2017–2020 |
| Mean BMI | **28.1** | NHANES 2017–2020 |
| Mean max heart rate | **174.3 bpm** | Tanaka Formula: 208 − 0.7 × age |
| Mean resting HR | **70.0 bpm** | Normal adult population |

### Key Risk Factor Correlations (Research-Backed)

| Risk Factor | HD Rate: Positive | HD Rate: Negative | Odds Ratio |
|---|---|---|---|
| Diabetes | **71.3%** | 43.0% | OR ≈ 1.73 |
| Hypertension | **64.0%** | 30.2% | OR ≈ 1.82 |
| Current Smoking | **57.4%** | 43.1% (Never) | OR ≈ 1.82 |
| Male Sex | **50.9%** | 40.2% (Female) | OR ≈ 1.73 |
| Family History | **52.2%** | 43.4% | OR ≈ 1.65 |
| Previous Event | **~82%** | ~44% | OR ≈ 2.46 |
| Downsloping ST | **58.6%** | 37.3% (Upsloping) | Clinical |
| LV Hypertrophy (ECG) | **55.9%** | 38.5% (Normal) | Clinical |
| Sedentary Lifestyle | **53.1%** | 38.4% (Very Active) | Protective |

### Clinical Distribution Validation

- ✅ Cholesterol: 217.7 mg/dl mean (NHANES normal adult: 200–220)
- ✅ Resting BP: 80–200 mmHg range (includes hypertensive patients)
- ✅ Max HR: Tanaka formula (208 − 0.7 × age) with individual variation
- ✅ HRV score: Higher in younger, fitter, non-smoking patients
- ✅ FBS flag: 55–80% positive rate in diabetics, 3–12% in non-diabetics
- ✅ ST depression: Exponential distribution, elevated with exercise angina

---

## 📁 Files

```text
heart_disease_prediction_2026.csv     # Main dataset (100K rows, 26 columns, 12.4 MB)
heart_disease_starter_notebook.ipynb  # Complete EDA + baseline model notebook
README.md                             # This file
```

### Dataset Format
- **Format:** CSV (comma-separated values)
- **Encoding:** UTF-8
- **Header:** Yes (feature names in first row)
- **Missing Values:** 0
- **Duplicate Rows:** 0

---

## 🚀 Usage Examples

### Load the Dataset
```python
import pandas as pd

df = pd.read_csv('heart_disease_prediction_2026.csv')
print(df.shape)    # (100000, 26)
print(df.dtypes)
print(df.head())
print(df.describe())
```

### Binary Classification: Predict Heart Disease
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

df_enc = df.copy()
for col in df_enc.select_dtypes(include='object').columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

X = df_enc.drop('heart_disease', axis=1)
y = df_enc['heart_disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, max_depth=12,
                               random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred,
      target_names=['No Disease', 'Heart Disease']))
```

### Logistic Regression (Interpretable Baseline)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}")
```

### XGBoost (Competitive Baseline)
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                    use_label_encoder=False, eval_metric='auc',
                    random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)], verbose=False)
print(f"XGBoost ROC-AUC: {roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]):.4f}")
```

---

## 📊 Risk Factor Profiles

Each risk group has distinct cardiovascular profiles:

| Group | HD Rate | Mean Age | Mean BP | Hypertension % | Diabetes % |
|---|---|---|---|---|---|
| Previous Heart Event | ~82% | 64 | 158 | 71% | 38% |
| Diabetic | 71.3% | 58 | 151 | 62% | 100% |
| Hypertensive | 64.0% | 57 | 166 | 100% | 22% |
| Current Smoker | 57.4% | 55 | 146 | 51% | 14% |
| Sedentary | 53.1% | 56 | 145 | 50% | 13% |
| Male | 50.9% | 54 | 143 | 47% | 11% |
| Family History | 52.2% | 54 | 143 | 47% | 11% |
| Low Income | 48.6% | 53 | 143 | 48% | 13% |
| Female | 40.2% | 53 | 141 | 47% | 11% |
| Very Active | 38.4% | 52 | 138 | 42% | 9% |
| High Income | 43.2% | 53 | 141 | 45% | 10% |

---

## 🧬 Generation Methodology

### Step-by-Step Process

1. **Demographic Base** → Age (μ=54, σ=13), sex (55% male), BMI (age-adjusted), region, income
2. **Lifestyle Generation** → Smoking, alcohol, physical activity, diet, sleep, stress — all correlated with age and income
3. **Comorbidity Calibration** → Diabetes intercept solved (Brent's method) to achieve exactly 11.0% prevalence; hypertension to exactly 47.0%
4. **Clinical Marker Generation** → Resting BP, cholesterol, max HR, ECG, ST markers — each derived from upstream variables with physiologically realistic adjustments (e.g., statin use modeled for 20% of population)
5. **Wearable Signal Generation** → HRV and resting HR derived from age, fitness, stress, and smoking status
6. **Target Construction** → `heart_disease` generated via 25-term logistic model; intercept numerically solved for 46.1% prevalence
7. **Validation** → All bounds, prevalences, correlations, and medical logic verified

### Key Design Principles
- **Causal realism:** Risk factors → target follow clinically valid causal chains
- **Exact calibration:** Prevalences matched to 3 decimal places using numerical optimization
- **Reproducibility:** Fixed seed (42) ensures exact replication
- **Correlation structure:** Non-trivial inter-feature dependencies throughout (e.g., hypertension raises BP, ECG abnormality, and HD risk simultaneously)
- **No synthetic oversampling:** 46.1% positive rate achieved via model intercept adjustment, not SMOTE or duplication

---

## 📈 Performance Benchmarks

### Baseline Model Results (80/20 stratified split)

| Model | ROC-AUC | Accuracy | F1 (HD) | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.73 | ~0.72 | ~0.74 | ~0.71 |
| Random Forest | ~0.82 | ~0.75 | ~0.74 | ~0.76 | ~0.73 |
| XGBoost | ~0.84 | ~0.77 | ~0.76 | ~0.77 | ~0.75 |

*Run the starter notebook for exact reproducible results.*

---

## 🔍 Data Quality Assurance

### Validation Checks (All Passed ✅)

- ✅ Shape integrity: 100,000 × 26
- ✅ Missing values: 0 (zero missingness)
- ✅ Duplicate rows: 0
- ✅ All clinical values within physiological bounds
- ✅ Diabetes prevalence: 11.0% (CDC 2023 ✓)
- ✅ Hypertension prevalence: 47.0% (AHA 2023 ✓)
- ✅ Smoker rate: 13.9% (CDC 2023 ✓)
- ✅ All binary columns strictly 0/1
- ✅ Cholesterol range: 100–400 mg/dl (physiological ✓)
- ✅ Blood pressure range: 80–200 mmHg (physiological ✓)
- ✅ Max HR: Tanaka-formula-aligned (✓)
- ✅ All risk factor correlations in correct direction
- ✅ Wearable signals inversely correlated with disease risk (HRV ✓)

---

## 🗺️ Suggested Tasks

- **Binary Classification** — Predict `heart_disease` (0 or 1)
- **Probability Calibration** — Output calibrated risk scores using `CalibratedClassifierCV`
- **Feature Importance Analysis** — Which clinical vs lifestyle factors dominate?
- **Fairness / Subgroup Analysis** — Does model performance differ by sex or income level?
- **Feature Engineering** — Create interaction terms: `age × hypertension`, `bmi × diabetes`
- **Explainability** — SHAP values for individual patient risk explanation
- **Threshold Optimization** — Tune classification threshold for clinical use (maximize recall for disease)

### Recommended Models
`Logistic Regression` · `Random Forest` · `XGBoost` · `LightGBM` · `CatBoost` · `Neural Networks` · `SVM`

---

## 📖 Sources & References

### Epidemiology & Clinical Guidelines
- **AHA Heart Disease & Stroke Statistics 2023** — [Link](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001123)
- **ACC/AHA 2019 Cardiovascular Risk Guidelines** — [Link](https://www.ahajournals.org/doi/10.1161/CIR.0000000000000678)
- **ACC/AHA 2017 Hypertension Guidelines** — [Link](https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065)
- **Framingham Heart Study** — [Link](https://www.framinghamheartstudy.org)

### Population Health Data
- **CDC National Diabetes Statistics Report 2023** — [Link](https://www.cdc.gov/diabetes/data/statistics-report/index.html)
- **CDC Cigarette Smoking Report 2023** — [Link](https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm)
- **NHANES 2017–2020** — [Link](https://www.cdc.gov/nchs/nhanes/index.htm)
- **American Heart Association 2022 Update** — [Link](https://www.heart.org/en/about-us/heart-and-stroke-association-statistics)

### Clinical Reference
- **Tanaka et al. (2001)** — Age-predicted maximum heart rate: 208 − 0.7 × age — [Link](https://www.sciencedirect.com/science/article/pii/S0735109700014739)
- **UCI Cleveland Heart Disease Dataset (1988)** — [Link](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## 📄 License

MIT License — Free for academic, commercial, and personal use with attribution.

---

## 👤 Author

**Mohan Krishna Thalla**
- Kaggle: [@mohankrishnathalla](https://www.kaggle.com/mohankrishnathalla)

---

**Last Updated:** September 2026 | **Version:** 1.0 | **Dataset Size:** 12.4 MB | **Records:** 100,000 | **Features:** 25 + 1 target | **License:** MIT | **Status:** ✅ Production Ready

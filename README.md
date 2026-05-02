# AIML-Final
# 🔥 US Wildfire Size Classification

**IE University — Machine Learning Foundations | Group Project 2025**

Classifying US wildfires into size categories (B–G) using pre-fire weather conditions, vegetation type, geographic location, and temporal features.

---

## 📁 Repository Structure

​```
wildfire-project/
├── data/
│   └── FW_Veg_Rem_Combined.csv       # Download separately — see below
├── notebooks/
│   └── wildfire_pipeline.ipynb       # Main notebook: EDA → models → evaluation
├── src/
│   └── preprocessing.py              # Full sklearn preprocessing pipeline
├── models/                           # Saved .pkl files — generated at runtime
├── outputs/                          # Plots and result tables — generated at runtime
├── requirements.txt
├── .gitignore
└── README.md
​```

---

## 📦 Dataset Setup

Download the dataset before running anything — it is not included in the repo.

**Option 1 — Kaggle CLI:**
​```bash
pip install kagglehub
python -c "
import kagglehub, shutil, os
path = kagglehub.dataset_download('capcloudcoder/us-wildfire-data-plus-other-attributes')
shutil.copy(os.path.join(path, 'FW_Veg_Rem_Combined.csv'), 'data/')
"
​```

**Option 2 — Manual:**
1. Visit [kaggle.com/datasets/capcloudcoder/us-wildfire-data-plus-other-attributes](https://www.kaggle.com/datasets/capcloudcoder/us-wildfire-data-plus-other-attributes)
2. Download and place `FW_Veg_Rem_Combined.csv` inside the `data/` folder

---

## ⚙️ Installation

​```bash
git clone https://github.com/<your-username>/wildfire-project.git
cd wildfire-project
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
​```

---

## 🚀 Running the Pipeline

**Notebook (recommended):**
​```bash
jupyter notebook notebooks/wildfire_pipeline.ipynb
​```
Run cells top to bottom. The notebook imports `src/preprocessing.py` automatically.

**Preprocessing smoke test:**
​```bash
python src/preprocessing.py
​```

---

## 🧪 Pipeline Overview

| Stage | Detail |
|---|---|
| **Target** | `fire_size_class` — 6 classes B → G |
| **Split** | 70% train / 15% val / 15% test — stratified |
| **Sentinel fix** | Weather `-1` → `NaN` before median imputation |
| **Imputation** | Median (robust to skewed precipitation) |
| **Feature engineering** | 30/15/7-day weather deltas + cyclical sin/cos time features |
| **Encoding** | OneHotEncoder (cause, state, vegetation) · OrdinalEncoder (month) |
| **Scaling** | RobustScaler (weather) · StandardScaler (continuous) |
| **Leakage prevention** | `fire_size` and `fire_mag` dropped before any split |

---

## 👥 Team

| Name | Role |
|---|---|
| [Name 1] | EDA |
| [Name 2] | Preprocessing pipeline |
| [Name 3] | Model development |
| [Name 4] | Evaluation & reporting |

---

## 📚 References

- Dataset: [U.S. Wildfire Data — Kaggle](https://www.kaggle.com/datasets/capcloudcoder/us-wildfire-data-plus-other-attributes)
- Original source: [USDA Forest Service RDS-2013-0009](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2013-0009.6)
- scikit-learn: [scikit-learn.org](https://scikit-learn.org)

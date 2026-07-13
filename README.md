# breast-cancer-diagnostics
# Clinical-Grade Breast Cancer Diagnostic Pipeline using XGBoost

An end-to-end machine learning and feature engineering pipeline designed to classify breast cancer tumors as Malignant or Benign using digitized fine needle aspirate (FNA) cellular metrics. This project optimizes an XGBoost classifier via a parallelized Grid Search and implements post-prediction threshold tuning to minimize critical clinical errors.

## Key Results
* **Cross-Validation ROC-AUC:** 99.18%
* **Validation Set ROC-AUC:** 99.17%
* **Optimized Decision Threshold:** 0.75 (Balanced optimization reducing False Negatives without ballooning False Alarms)

.

## Project Architecture & Workflow

### 1. Feature Engineering
To enhance the baseline dataset, two custom biological interaction metrics were engineered:
* `radius_texture_product`: Captures the combined risk profile of extreme cell size overlapping with dense tissue irregularities.
* `compactness_concavity_ratio`: Evaluates the proportionality of cellular deformities to structural compactness.
* *Result:* Both engineered features successfully ranked within the **Top 10 Most Influential Biomarkers** during feature importance evaluation.

### 2. Automated Hyperparameter Tuning
Instead of manual parameter selection, an automated optimization engine was constructed using `GridSearchCV` with 5-fold cross-validation, scanning 27 unique parameter combinations (135 total model training iterations running in parallel via `n_jobs=-1`).

**Winning Hyperparameters:**
* `learning_rate`: 0.05
* `max_depth`: 3 (Shallower depth prioritized to prevent overfitting)
* `n_estimators`: 100

### 3. Clinical Threshold Optimization
In cancer diagnostics, **False Negatives** (missing a malignant tumor) are catastrophic. A standard default classification boundary of `0.50` was dynamically adjusted using a data-driven Precision-Recall trade-off analysis. Shifting the boundary to a cautious **`0.75` threshold** optimized the clinical metrics cleanly:

| Metric State | Default Threshold (0.50) | Balanced Threshold (0.75) |
| :--- | :---: | :---: |
| **False Negatives (Missed Cancers)** | 4 | **3** |
| **False Positives (False Alarms)** | 4 | **4** |

---

## Tech Stack & Tools Used
* **Language:** Python
* **Core ML Framework:** Scikit-Learn, XGBoost
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebooks


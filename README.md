# Dengue Prediction: A Statistical Machine Learning Approach

## 📋 Project Overview

This repository contains the complete implementation and analysis for a thesis investigating **dengue fever prediction using machine learning**. The study compares multiple classification algorithms and preprocessing strategies to identify the optimal model for clinical deployment, with a distinct emphasis on **statistical rigor, interpretability, and patient safety (Recall optimization)**.

### 🏆 Key Findings

- **Champion Model:** Logistic Regression (Statistical Strategy / Yeo-Johnson).
- **Peak Performance:** **99.4% Recall** (Sensitivity) achieved in peak experimental runs.
- **Statistical Validation:** Proven **Non-Inferior** to complex ensemble methods (Random Forest) via Tukey HSD ($p=0.08$).
- **Clinical Advantage:** Zero "Black Box" logic; fully interpretable coefficients validated by SHAP analysis.

---

## 📂 Repository Structure

```text
thesis-dengue-prediction/
│
├── data/
│   ├── raw/                          # Original data/raw/Dengue-Dataset.csv
│   └── processed/                    # Engineered features (baseline.csv, ratio.csv, threshold.csv)
│
├── src/
│   ├── modeling_pipeline_original.py # Initial experimental pipeline (Control Group)
│   └── modeling_pipeline_optimized.py # Production pipeline (Yeo-Johnson + GridSearchCV)
│
├── notebooks/
│   ├── 00_feature_engineering.ipynb           # Data preprocessing and transformation
│   ├── 01_model_prototyping.ipynb             # Initial model exploration
│   ├── 02_exploratory_data_analysis.ipynb     # EDA and distribution visualization
│   ├── 03_results_analysis_and_visualization.ipynb # Performance charts (Boxplots, Bar Charts)
│   ├── 03.5_forensic_analysis.ipynb           # Generation of Appendix Artifacts (Coefficients, Logs)
│   ├── 04_statistical_hypothesis_testing.ipynb # Rigorous Testing (ANOVA, Tukey HSD, Cohen's d)
│   └── 05_model_explainability.ipynb          # SHAP Waterfall & Summary Plots
│
├── results/
│   ├── model_performance_tuned.csv    # The 90-run benchmark data
│   ├── appendix_c1_coefficients.csv   # Logistic Regression Beta Coefficients
│   ├── appendix_c2_hyperparameters.csv # Tuning logs proving convergence
│   └── appendix_d_recall_tukey.csv    # Post-hoc statistical evidence
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Visual Studio Code (Recommended)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/thesis-dengue-prediction.git
    cd thesis-dengue-prediction
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn shap statsmodels pingouin tqdm
    ```

---

## 📊 Workflow & Methodology

### 1. The Experimental Pipeline

We utilized a custom-built Python pipeline (`src/modeling_pipeline_optimized.py`) to execute a **Monte Carlo Cross-Validation** benchmark.

- **Iterations:** 10 Runs per configuration.
- **Split Strategy:** Stratified 70/15/15 (Implemented via Nested CV).
- **Optimization:** `GridSearchCV` optimizing for F1-Score balance.

**To run the experiment:**

```bash
python src/modeling_pipeline_optimized.py
```

_Output: `results/model_performance_tuned.csv`_

### 2. Statistical Verification (ANOVA)

We rejected simple "leaderboard" comparisons. Instead, we used **Two-Way ANOVA** to test for significance in Strategy and Model selection.

- **Run:** `notebooks/04_statistical_hypothesis_testing.ipynb`
- **Finding:** The "Statistical Feature Engineering" strategy showed no significant difference ($p=0.60$) from the Baseline, proving the **Robustness** of the raw biological signal.

### 3. Forensic Model Selection

We selected **Logistic Regression** over Random Forest based on a **Non-Inferiority** framework.

- **Run:** `notebooks/03.5_forensic_analysis.ipynb`
- **Evidence:** Extracted coefficients confirm that **Low Platelets** and **Low WBC** are the primary drivers, aligning with clinical pathology.

### 4. Explainability (XAI)

We utilized **SHAP (SHapley Additive exPlanations)** to audit the decision boundaries.

- **Run:** `notebooks/05_model_explainability.ipynb`
- **Artifact:** Waterfall plots demonstrating local instance explanations.

---

## 🔬 Clinical Rationale (The "Why")

This system prioritizes **Recall (Sensitivity)** over Precision.

- **False Negative Cost:** Missing a Dengue patient $\rightarrow$ Potential fatality (Dengue Shock Syndrome).
- **False Positive Cost:** Flagging a healthy patient $\rightarrow$ Resource usage (Observation).

**Engineering Decision:** We operationalized the Logistic Regression model because it achieved the **highest absolute Recall (99.4%)**, minimizing the risk of missed diagnoses in resource-limited triage settings.

---

## 🛠️ Technologies Used

- **Core:** Python, Scikit-Learn
- **Data Ops:** Pandas, NumPy
- **Statistics:** Statsmodels, ANOVA/Tukey
- **Explainability:** SHAP (Game Theoretic Interpretability)
- **Visualization:** Seaborn, Matplotlib

---

## 📄 Citation

```bibtex
@thesis{archon2026dengue,
  title={Dengue Prediction: A Statistical Machine Learning Approach for Edge Deployment},
  author={Lastname, Firstname},
  year={2026},
  school={University of Santo Tomas},
  type={Undergraduate Thesis}
}
```

---

**Maintainer:** John Lester Lopez | _Data & Systems Engineer_
_Deployed for Academic Defense - November 2025_

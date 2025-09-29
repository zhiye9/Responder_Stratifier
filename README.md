# Responder Stratifier

A machine learning framework for identifying intervention responders using both **Causal Forest Double Machine Learning (DML)** and **tree-based ensemble models (Random Forest and XGBoost)** to predict clinical outcomes based on multi-omics data from the HPP ([Human Phenotype Project](https://humanphenotypeproject.org/)) dataset.

## Overview
<img width="3234" height="2107" alt="EIC_causal_forest" src="https://github.com/user-attachments/assets/5c30108d-0bae-4dd3-ae5d-a6c610a20c7a" />

This project provides two complementary approaches for responder stratification:

1. **Prediction of urate using multi-omics data**: Benchmarks performance using Random Forest and XGBoost.
2. **Causal inference of intervention**: Estimates heterogeneous treatment effects (CATE) for possible intervention using Causal Forest.

The first approach targets blood urate level prediction using multi-omics features, while the second model identifies subjects who will respond to specific interventions.

### Key Features

- **Outcome**: Blood urate levels (mg/dL)
- **Treatment/Features**: Multi-omics data including:
  - Anthropometric measures (sex, age, BMI, waist circumference)
  - Blood biomarkers (creatinine, albumin, ALT, etc.)
  - Microbiome PCoA
  - Dietary intake proportions
  - SNP genetic variants
- **Target**: Any intervention feature

## Project Structure

```
Responder_Stratifier/
├── README.md                           # Project documentation
├── Responder_classifier.ipynb          # Causal Forest inference
├── scr/
│   └── Nested_CV_Prediction.py        # ML benchmarking with nested CV
├── data/                               # Data directory (privacy)
├── models/                             # Model storage (to be downloaded)
└── Responder_Stratifier_report.pdf    # Detailed methodology report
```

## Methodology

### 1. Nested Cross-Validation ML (`scr/Nested_CV_Prediction.py`)

**Purpose**: Benchmark prediction performance and feature importance

- **Models**: Random Forest and XGBoost
- **Validation**: 5-fold nested cross-validation
- **Target**: Blood urate levels
- **Metrics**: R² scores with comprehensive hyperparameter tuning

### 2. Causal Forest DML Approach (`Responder_classifier.ipynb`)

**Purpose**: Estimate individual treatment effects for responder stratification

- **Framework**: Double Machine Learning with Causal Forest
- **Target**: CATE estimation in mg/dL per unit feature change
- **Responder Definition**: Subjects whose CATE meets clinical thresholds

```python
# Causal Forest Configuration
cf = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=600, max_depth=10),
    model_t=RandomForestRegressor(n_estimators=600, max_depth=10),
    n_estimators=900, min_samples_leaf=10, cv=5
)
```


## Usage

### Training Pipeline

The training process uses HPP dataset (data not included due to privacy constraints):

1. Load training data with outcome, treatment (any multi-omics feature), and covariates
2. Fit Causal Forest DML model
3. Save trained model for inference


### Nested CV Benchmarking

```python
# Run comprehensive ML evaluation
python3 scr/Nested_CV_Prediction.py

# Outputs:
# - Model performance metrics
# - Feature importance rankings
```

### Inference Pipeline

```python
# 1. Training
df = pd.read_csv("urtate_data.csv")
target = "Vegetables"  # or any feature
X_train = df.drop(columns=[target, "bt__urate_float_value"])
y_train = df["bt__urate_float_value"]
T_train = df[target]

cf.fit(y_train, T_train, X=X_train)

# 2. Inference
cate_hat = cf.effect(X_test)
ate_hat = cf.ate(X_test)

# 3. Responder Classification
clinic_target = -0.3  # mg/dL reduction
intervention = 1      # unit increase
threshold = clinic_target / intervention
responders = (cate_hat < threshold)
```


## Key Outputs

### Causal Forest DML
- **Individual Treatment Effects (CATE)**: Personalized effect estimates
- **Average Treatment Effect (ATE)**: Population-level effects
- **Responder Classification**: Binary assignment with clinical thresholds

### Nested CV ML
- **Performance Benchmarks**: R² scores across folds
- **Feature Importance**: Gini and gain-based rankings
- **Model Comparison**: Random Forest vs XGBoost performance

## Requirements

```python
numpy
pandas
scikit-learn
econml
joblib
scipy
```

## Data Privacy

Training data from the Human Phenotype Project is not included in this repository due to data privacy and usage agreements. Please contact HPP ([Human Phenotype Project](https://humanphenotypeproject.org/)) for data access. The trained model is provided for inference on new subjects with similar feature profiles.


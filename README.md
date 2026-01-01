# 🚀 Spaceship Titanic — Interpretable ML Pipeline & Deployment (CatBoost)

## Overview

This project tackles Kaggle’s **Spaceship Titanic** classification problem using a **first-principles, leakage-aware machine learning pipeline** built around a `CatBoostClassifier`.

Rather than chasing leaderboard tricks or heavy ensembling, the focus is on **robust preprocessing, principled modeling, interpretability, and deployment readiness**.  
The final solution achieves a competitive public leaderboard score while remaining **explainable, reproducible, and production-compatible**.

---

## Problem Statement

Passengers aboard the *Spaceship Titanic* were transported to an alternate dimension due to a spacetime anomaly.  
Given structured passenger data, the task is to predict whether a passenger was **Transported** (`True` / `False`).

---

## Key Design Goals

- Prevent data leakage (temporal & proxy features)
- Apply log transforms only where statistically justified
- Build a reproducible and inference-safe pipeline
- Prioritize interpretability over brute-force ensembling
- Enable real-world deployment via a prediction API

---

## Feature Engineering

- Domain-aware feature grouping (e.g. spending behavior, demographics)
- Explicit leakage removal (temporal and transaction-derived fields)
- Custom preprocessing logic for:
  - Log-scaling skewed numerical features
  - Robust missing value handling
  - Consistent transformations applied during both training and inference to avoid train–serve skew

---

## Modeling Approach

**CatBoostClassifier** as the primary model:

- Handles non-linear feature interactions effectively
- Robust to feature scaling
- Strong performance on structured/tabular data

Nested cross-validation used for:

- Hyperparameter tuning
- Honest generalization estimation

Final CatBoost model retrained on the full dataset after validation.

This approach balances **performance, stability, and interpretability** without overfitting to leaderboard noise.

---

## Interpretability & Diagnostics

Model behavior is analyzed using:

- SHAP values (global and local explanations)
- Feature importance validation
- Interaction-aware reasoning enabled by tree-based structure
- Detection of misleading or proxy features

This ensures predictions are **auditable, explainable, and trustworthy**, even in deployment settings.

---

## Deployment

- Trained model and preprocessing artifacts saved using `joblib`
- FastAPI-based prediction service
- Typed input schema to enforce safe and consistent inference
- Preprocessing and model bundled together to ensure deployment parity with training

---

## Results

- **Public Kaggle Score:** ~0.79
- Competitive performance without leakage or heavy ensembling
- Fully interpretable, deployment-ready ML system

---

## Project Philosophy

This project emphasizes real ML engineering, not leaderboard gaming:

> **Understand the data → model responsibly → explain predictions → deploy safely**

---

## Tech Stack

- Python  
- NumPy
- Pandas 
- CatBoost  
- Scikit-learn (validation & utilities)  
- SHAP  
- FastAPI  
- Joblib  
- Kaggle API  

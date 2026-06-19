# Practical Application III — Comparing Classifiers

**Author:** Sravanthi Gandu
**Course:** UC Berkeley ML/AI — Module 17 Required Assignment 17.1

## Overview

This project compares four classifiers — **K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines** — on a real marketing dataset from a Portuguese bank. The data comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and records the outcomes of **17 telemarketing campaigns** (May 2008 – Nov 2010, 41,188 contacts) selling long-term **term deposits**.

📓 **Notebook:** [`prompt_III.ipynb`](prompt_III.ipynb)

## Business Objective

Predict, **before a call is placed**, which clients are most likely to subscribe to a term deposit, using attributes the bank already knows. A good model lets the marketing team prioritize high-probability prospects to **raise the conversion rate, cut wasted call cost, and improve customer experience**.

## Data & Approach

- **Features used:** the 7 *bank-client* attributes — `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` (per the assignment prompt).
- **Target:** `y` — did the client subscribe? (encoded yes→1, no→0).
- **Data quality:** no explicit `NaN`s; missing values are coded as the string `'unknown'` and kept as their own category. The `duration` feature is **excluded** because it is leakage (only known after a call ends).
- **Preprocessing:** `StandardScaler` on `age`, `OneHotEncoder` on categoricals, wrapped in an sklearn `Pipeline` (28 encoded features).
- **Split:** 80/20 stratified train/test.

## Key Results

The classes are heavily imbalanced (**only ~11.3% subscribed**), so a "always predict no" baseline already scores **88.7% accuracy**. Accuracy is therefore misleading, and tuning was optimized for **ROC-AUC** with `class_weight='balanced'`.

**Default models (accuracy):** all cluster around the ~88–89% baseline; the Decision Tree overfits and SVM is ~500× slower to train than Logistic Regression.

**Tuned models (GridSearchCV, scored on ROC-AUC):**

| Model | Best Params | CV ROC-AUC | Test ROC-AUC | Recall (yes) |
| ----- | ----------- | ---------- | ------------ | ------------ |
| Logistic Regression | C=0.01 | 0.650 | 0.650 | 0.64 |
| KNN | n_neighbors=31, distance | 0.621 | 0.634 | 0.03 |
| Decision Tree | max_depth=7, min_samples_leaf | 0.647 | **0.659** | 0.51 |
| SVM | C=0.1, rbf | 0.650 | 0.653 | 0.60 |

All tuned models beat random chance (0.50). The **Decision Tree** gave the best test ROC-AUC (~0.66) and the **balanced Logistic Regression** gave the best recall on subscribers while being far faster — a strong practical choice.

## Findings & Recommendations

- **Demographics alone carry weak signal** — these 7 features push ROC-AUC only into the ~0.63–0.66 range.
- **High-converting segments:** *students*, *retired* clients, and clients with a *university degree* subscribe well above average; *blue-collar*, *services*, and *entrepreneur* clients below average. Targeting the former immediately improves efficiency.
- **Next steps:**
  1. Add campaign + macro-economic features (`contact`, `month`, `poutcome`, `euribor3m`, `nr.employed`, …, still excluding `duration`) — the highest-leverage improvement.
  2. Choose the probability threshold by **expected profit** (call cost vs. subscription revenue), not the default 0.5.
  3. Try ensemble models (Random Forest, Gradient Boosting/XGBoost).
  4. Operationalize as a **lead-scoring list** worked top-down; retrain as economic conditions shift.

## Repository Structure

```
module17_starter/
├── README.md                 # this file
├── prompt_III.ipynb          # full analysis notebook
├── CRISP-DM-BANK.pdf         # accompanying paper
└── data/
    ├── bank-additional-full.csv   # 41,188 records (used)
    ├── bank-additional.csv        # 10% sample
    └── bank-additional-names.txt  # data dictionary
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook prompt_III.ipynb   # Run All
```

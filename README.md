# From Data to Dollars — Predicting Insurance Charges

## Overview

This project builds a regression model that predicts individual medical costs billed by a health insurance company. The goal is to help the insurer tailor services and help customers plan healthcare expenses more effectively.

The model is trained on `insurance.csv` and then applied to unseen customers in `validation_dataset.csv`.

## Dataset

### `insurance.csv` (training data)

| Column | Type | Description |
|---|---|---|
| `age` | int | Age of the primary beneficiary |
| `sex` | object | Gender of the insurance contractor |
| `bmi` | float | Body mass index |
| `children` | int | Number of dependents covered by the plan |
| `smoker` | object | Whether the beneficiary smokes (yes/no) |
| `region` | object | Residential area in the US (4 regions) |
| `charges` | float | Individual medical costs billed by insurance |

### `validation_dataset.csv`

Same features as the training data but **without** the `charges` column. Used to test the model on unseen customers.

## Objectives

1. Clean and preprocess the insurance dataset.
2. Train a regression model to predict `charges`.
3. Evaluate the model using the **R-Squared Score** (must exceed `0.65`).
4. Predict charges for `validation_dataset.csv` and store them in a new column called `predicted_charges`.
5. Enforce a minimum basic charge of `1000`.

## Approach

### 1. Data cleaning
- Removed rows with missing values.
- Stripped `$` and `,` characters from `charges` and cast to `float`.
- Encoded `sex` and `smoker` as binary integers.
- Lowercased `region` and applied one-hot encoding with `drop_first=True`.

### 2. Preprocessing as a reusable function
All transformations are wrapped in a `preprocess()` function so that the training and validation datasets go through **exactly the same pipeline** — a common source of bugs in production ML.

### 3. Model training
Two models were compared:
- **Linear Regression** — interpretable baseline.
- **Random Forest Regressor** — captures non-linearities.

### 4. Evaluation
Models were evaluated with the R-Squared Score on a held-out test set (20%).

### 5. Prediction
The best model was applied to `validation_dataset.csv`, and predictions were floored at `1000`.

## Results

| Model | R² Score |
|---|---|
| Linear Regression | **0.767** |
| Random Forest Regressor | (lower or similar) |

**Final model:** Linear Regression — best balance of accuracy and interpretability.

### Top predictive features (coefficients)

| Feature | Coefficient |
|---|---|
| `smoker` | ~24,110 |
| `region_northwest` | ~535 |
| `children` | ~316 |
| `bmi` | ~295 |
| `age` | ~266 |
| `sex` | ~153 |
| `region_southwest` | ~58 |
| `region_southeast` | ~ -148 |

Smoking status is by far the strongest driver of insurance charges.

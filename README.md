# Student Dropout Predictor

> A machine learning pipeline that predicts whether a student is at risk of dropping out — so schools can intervene early, before it's too late.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Library](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-complete-green)

---

## What it does

This project trains a classification model on student academic and demographic data to predict dropout risk. The output is a probability score per student, letting educators prioritize who needs support first.

## Why this matters

Dropout is one of the most expensive and preventable problems in education — especially in developing economies. Predicting it early turns a reactive system (losing students) into a proactive one (supporting them). The goal of this project isn't just accuracy — it's building a pipeline a real school could actually use.

## Pipeline

```
Raw data → Cleaning → Feature engineering → Train/test split
                                                 ↓
                     Evaluation ← Model training (multiple classifiers)
                          ↓
                     Best model → Predictions + probability scores
```

## Tech stack

- **Python 3.10+**
- **pandas** — data wrangling
- **scikit-learn** — modeling (Logistic Regression, Random Forest, etc.)
- **matplotlib / seaborn** — EDA and evaluation plots
- **Jupyter** — iteration and documentation

## Key steps

1. **EDA** — understanding distributions, missing values, target imbalance
2. **Feature engineering** — encoding categoricals, scaling, handling missing data
3. **Model comparison** — testing multiple classifiers and picking the best by F1 / ROC-AUC
4. **Evaluation** — confusion matrix, classification report, ROC curve
5. **Interpretation** — which features drive the prediction (so humans can act on it)

## Running it

```bash
git clone https://github.com/Iyimoga/Students-Dropout-Predictor.git
cd Students-Dropout-Predictor
pip install -r requirements.txt
jupyter notebook
```

Open the main notebook and run cells top to bottom.

## Results

Random Forest reached 90% accuracy and 0.82 F1 on the test set

## What I learned

- Feature engineering usually beats model choice — time spent on data quality compounds
- Imbalanced classification needs the right metric; accuracy alone lies
- A model is only useful if someone non-technical can act on its output

## About

Built by **[Iyimoga Joseph Nana](https://github.com/Iyimoga)** as part of ongoing ML coursework and self-study.

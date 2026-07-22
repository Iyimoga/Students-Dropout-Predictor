"""
Train an ensemble dropout prediction model from Logistic Regression,
Support Vector Machine, and Random Forest.

This script keeps the project aligned with the revised topic:
"An Ensembled Machine Learning Model for Prediction of Students Dropout
in Nigerian Universities".
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "dataset"
TOP_N_FEATURES = 24


def get_positive_probability(model, x):
    return model.predict_proba(x)[:, 1]


def evaluate_model(name, model, x_train, y_train, x_test, y_test, cv):
    cv_accuracy = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy").mean()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = get_positive_probability(model, x_test)

    return {
        "Model": name,
        "CV Accuracy": cv_accuracy,
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def save_comparison_plot(comparison):
    metrics = ["Test Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = comparison.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(14, 7))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0.70, 1.00)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_bars.png", dpi=200)
    plt.close()


def save_roc_plot(trained_models, x_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in trained_models.items():
        y_proba = get_positive_probability(model, x_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.title("ROC Curves - Model Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=200)
    plt.close()


def save_confusion_matrices(trained_models, x_test, y_test):
    n_models = len(trained_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        cm = confusion_matrix(y_test, model.predict(x_test))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Graduate", "Dropout"],
            yticklabels=["Graduate", "Dropout"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=200)
    plt.close()


def save_mi_plot(mi_scores):
    top_scores = mi_scores.head(20).sort_values()
    plt.figure(figsize=(12, 9))
    top_scores.plot(kind="barh")
    plt.title("Top 20 Features by Mutual Information")
    plt.xlabel("Mutual Information Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mi_scores.png", dpi=200)
    plt.savefig(BASE_DIR / "dataset.png", dpi=200)
    plt.close()


def main():
    sns.set_style("whitegrid")

    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    df = df[df["Target"].isin(["Dropout", "Graduate"])].copy()

    y = df["Target"].map({"Graduate": 0, "Dropout": 1})
    x = df.drop(columns=["Target"])

    mi = mutual_info_classif(x, y, random_state=RANDOM_STATE)
    mi_scores = pd.Series(mi, index=x.columns).sort_values(ascending=False)
    features = mi_scores.head(TOP_N_FEATURES).index.tolist()
    x = x[features]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logistic_regression = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    support_vector_machine = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    base_models = {
        "Logistic Regression": logistic_regression,
        "Support Vector Machine": support_vector_machine,
        "Random Forest": random_forest,
    }

    ensemble_models = {
        "Soft Voting Ensemble": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ],
            voting="soft",
        ),
        "Stacking Ensemble": StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ],
            final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            stack_method="predict_proba",
            cv=5,
            n_jobs=-1,
        ),
    }

    models = {**base_models, **ensemble_models}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    trained_models = {}
    for name, model in models.items():
        result = evaluate_model(name, model, x_train_scaled, y_train, x_test_scaled, y_test, cv)
        results.append(result)
        trained_models[name] = model
        print(f"{name}: ROC-AUC={result['ROC-AUC']:.4f}, F1={result['F1-Score']:.4f}")

    comparison = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    best_name = comparison.iloc[0]["Model"]
    best_model = trained_models[best_name]

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    joblib.dump(comparison, BASE_DIR / "comparison_df.pkl")
    joblib.dump(trained_models, BASE_DIR / "all_models.pkl")
    joblib.dump(best_name, BASE_DIR / "best_model_name.pkl")
    joblib.dump(best_model, BASE_DIR / "dropout_model.pkl")
    joblib.dump(best_model, BASE_DIR / "ensemble_model.pkl")
    joblib.dump(scaler, BASE_DIR / "scaler.pkl")
    joblib.dump(features, BASE_DIR / "features.pkl")
    joblib.dump(mi_scores.head(TOP_N_FEATURES), BASE_DIR / "feat_stats.pkl")

    save_comparison_plot(comparison)
    save_roc_plot(trained_models, x_test_scaled, y_test)
    save_confusion_matrices(trained_models, x_test_scaled, y_test)
    save_mi_plot(mi_scores)

    print("\nBest model by ROC-AUC:", best_name)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

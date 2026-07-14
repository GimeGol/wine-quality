import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
DATA_PATH = Path("data") / "WineQT.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quality_bin"] = (df["quality"] >= 7).astype(int)
    return df


def basic_eda(df: pd.DataFrame) -> None:
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nClass balance:\n", df["quality_bin"].value_counts().sort_index())
    print("\nClass proportion:\n", df["quality_bin"].value_counts(normalize=True).sort_index())

    corr = df.drop(columns=["Id"]).corr(numeric_only=True)
    target_corr = corr["quality_bin"].sort_values(ascending=False)
    print("\nCorrelation with target:\n", target_corr)

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="quality_bin")
    plt.title("Distribution of quality_bin")
    plt.xlabel("quality_bin")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_balance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric_features
            )
        ],
        remainder="drop"
    )


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if proba is not None else np.nan,
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    return metrics, model


def run_cv(name: str, pipe, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
    print(f"{name} CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def main():
    df = load_data(DATA_PATH)
    df = create_target(df)

    basic_eda(df)

    X = df.drop(columns=["quality", "quality_bin", "Id"])
    y = df["quality_bin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    log_model = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    rf_model = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    run_cv("Logistic Regression", log_model, X, y)
    run_cv("Random Forest", rf_model, X, y)

    results = []

    for name, model in [
        ("Logistic Regression", log_model),
        ("Random Forest", rf_model)
    ]:
        metrics, fitted = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)

        if hasattr(fitted.named_steps["model"], "feature_importances_"):
            importances = pd.Series(
                fitted.named_steps["model"].feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            importances.to_csv(RESULTS_DIR / "feature_importances_random_forest.csv")

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    print("\nModel comparison:\n", results_df)

    best_model = rf_model.fit(X_train, y_train)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC curve - Random Forest")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve_random_forest.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

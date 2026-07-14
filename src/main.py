import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_data(path="../data/WineQT.csv"):
    df = pd.read_csv(path)
    return df


def prepare_data(df):
    df = df.copy()
    df["quality_bin"] = (df["quality"] >= 7).astype(int)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    X = df.drop(columns=["quality", "quality_bin"])
    y = df["quality_bin"]

    return df, X, y


def create_results_dir(path="../results"):
    os.makedirs(path, exist_ok=True)


def run_eda(df, results_path="../results"):
    create_results_dir(results_path)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.countplot(x="quality_bin", data=df, palette="viridis")
    plt.title("Distribuição da variável alvo binária")
    plt.xlabel("Classe")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(f"{results_path}/target_distribution.png")
    plt.close()

    df.hist(bins=20, figsize=(16, 12), color="steelblue", edgecolor="black")
    plt.suptitle("Distribuição das variáveis numéricas", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{results_path}/histograms.png")
    plt.close()

    corr = df.drop(columns=["quality"]).corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Mapa de calor das correlações")
    plt.tight_layout()
    plt.savefig(f"{results_path}/correlation_heatmap.png")
    plt.close()

    corr_target = corr["quality_bin"].sort_values(ascending=False)
    corr_target.to_csv(f"{results_path}/correlation_with_target.csv", header=True)

    return corr_target


def train_models(X_train, y_train):
    log_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000))
    ])

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return log_model, rf_model


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo": name,
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    print(f"\nModelo: {name}")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def run_cross_validation(X, y, log_model, rf_model):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    log_cv = cross_val_score(log_model, X, y, cv=cv, scoring="f1")
    rf_cv = cross_val_score(rf_model, X, y, cv=cv, scoring="f1")

    return {
        "Regressão Logística": log_cv.mean(),
        "Random Forest": rf_cv.mean()
    }


def save_roc_curve(log_model, rf_model, X_test, y_test, results_path="../results"):
    create_results_dir(results_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(log_model, X_test, y_test, ax=ax, name="Regressão Logística")
    RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax, name="Random Forest")
    plt.title("Curva ROC dos modelos")
    plt.tight_layout()
    plt.savefig(f"{results_path}/roc_curve.png")
    plt.close()


def save_feature_importance(rf_model, X, results_path="../results"):
    create_results_dir(results_path)

    feature_importance = pd.DataFrame({
        "Variável": X.columns,
        "Importância": rf_model.feature_importances_
    }).sort_values(by="Importância", ascending=False)

    feature_importance.to_csv(f"{results_path}/feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance.head(10),
        x="Importância",
        y="Variável",
        palette="viridis"
    )
    plt.title("Top 10 variáveis mais importantes - Random Forest")
    plt.tight_layout()
    plt.savefig(f"{results_path}/feature_importance.png")
    plt.close()

    return feature_importance


def main():
    df = load_data()
    df, X, y = prepare_data(df)

    print("Dimensões da base:", df.shape)
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())

    corr_target = run_eda(df)
    print("\nCorrelação com a variável alvo:")
    print(corr_target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    log_model, rf_model = train_models(X_train, y_train)

    results = []
    results.append(evaluate_model("Regressão Logística", log_model, X_test, y_test))
    results.append(evaluate_model("Random Forest", rf_model, X_test, y_test))

    results_df = pd.DataFrame(results)
    print("\nResumo das métricas:")
    print(results_df)

    create_results_dir()
    results_df.to_csv("../results/model_metrics.csv", index=False)

    cv_results = run_cross_validation(X, y, log_model, rf_model)
    cv_df = pd.DataFrame(list(cv_results.items()), columns=["Modelo", "F1 médio CV"])
    cv_df.to_csv("../results/cross_validation_results.csv", index=False)

    print("\nValidação cruzada:")
    print(cv_df)

    save_roc_curve(log_model, rf_model, X_test, y_test)
    feature_importance = save_feature_importance(rf_model, X)

    print("\nImportância das variáveis:")
    print(feature_importance.head(10))


if __name__ == "__main__":
    main()

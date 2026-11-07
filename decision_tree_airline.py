"""
Decision Tree classification for Invistico_Airline.csv.

Target: satisfaction (satisfied / dissatisfied).
Feature types are automatically detected from pandas dtypes.

Install:
    pip install pandas scikit-learn matplotlib

Run:
    python decision_tree_airline.py
    python decision_tree_airline.py --data "path/to/Invistico_Airline.csv"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an airline satisfaction Decision Tree."
    )
    parser.add_argument("--data", default="Invistico_Airline.csv",
                        help="Input CSV file.")
    parser.add_argument("--target", default="satisfaction",
                        help="Name of the target column.")
    parser.add_argument("--output-dir", default="decision_tree_output",
                        help="Directory where plots and importances are saved.")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and remove rows whose target is missing (they cannot be trained).
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Use --data to provide its location."
        )
    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Columns: {df.columns.tolist()}")

    df = df.dropna(subset=[args.target]).copy()
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Automatically identify numeric and categorical predictor columns.
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    other_features = [
        c for c in X.columns
        if c not in numeric_features and c not in categorical_features
    ]
    categorical_features.extend(other_features)

    print(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Target: {args.target} (classification)")
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("\nMissing feature values:\n", X.isna().sum()[lambda s: s > 0])

    # Each preprocessing step is fitted within the Pipeline, using training
    # folds only. This prevents information from the test set leaking into training.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # 80/20 holdout split; stratify preserves the target class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ])

    # A compact grid balances useful tuning with runtime on this 130k-row dataset.
    parameter_grid = {
        "classifier__max_depth": [3, 5, 8],
        "classifier__min_samples_leaf": [5, 10],
        "classifier__criterion": ["gini"],
    }
    search = GridSearchCV(
        pipeline,
        param_grid=parameter_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("\nTraining and tuning model...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("\nBest parameters:", search.best_params_)
    print(f"Best cross-validation weighted F1: {search.best_score_:.4f}")

    # Evaluate only after tuning, on the untouched test split.
    predictions = best_model.predict(X_test)
    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nPrecision, recall and F1-score:")
    print(classification_report(y_test, predictions, zero_division=0))

    classes = best_model.named_steps["classifier"].classes_
    matrix = confusion_matrix(y_test, predictions, labels=classes)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(matrix, index=classes, columns=classes))

    # Save confusion matrix image.
    ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=classes
    ).plot(cmap="Blues", values_format="d")
    plt.title("Airline Satisfaction — Confusion Matrix")
    plt.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Map transformed (one-hot encoded) columns to the tree's importance values.
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    fitted_tree = best_model.named_steps["classifier"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = pd.Series(
        fitted_tree.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    print("\nTop 20 feature importances:")
    print(importances.head(20).to_string())
    importance_path = output_dir / "feature_importances.csv"
    importances.rename("importance").to_csv(importance_path, header=True)

    # max_depth=3 limits only the displayed diagram, not the trained model.
    plt.figure(figsize=(24, 12))
    plot_tree(
        fitted_tree,
        feature_names=feature_names,
        class_names=[str(c) for c in fitted_tree.classes_],
        filled=True,
        rounded=True,
        proportion=True,
        max_depth=3,
        fontsize=7,
    )
    plt.title("Decision Tree — Airline Satisfaction (first 3 levels shown)")
    plt.tight_layout()
    tree_path = output_dir / "decision_tree.png"
    plt.savefig(tree_path, dpi=250, bbox_inches="tight")
    plt.close()

    print("\nSaved files:")
    print(f"  {tree_path.resolve()}")
    print(f"  {confusion_path.resolve()}")
    print(f"  {importance_path.resolve()}")


if __name__ == "__main__":
    main()

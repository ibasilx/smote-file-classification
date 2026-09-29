"""Run the project's baseline, SMOTE, and feature-selection experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED, TEST_SIZE, TARGET, N_FEATURES, SELECTED = 42, 0.2, "Class", 512, 256
FILE_TYPES = ["csv", "dbase3", "doc", "eps", "gif", "gz", "html", "jpg", "kmz", "log", "pdf", "png", "ppt", "ps", "swf", "text", "txt", "unk", "xls", "xml"]


def models():
    return [RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), SVC(kernel="rbf", C=1.0, gamma="scale"), DecisionTreeClassifier(random_state=SEED), AdaBoostClassifier(n_estimators=100, random_state=SEED), MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=SEED)]


def macro_fpr(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = matrix.sum()
    values = []
    for i in range(len(labels)):
        fp = matrix[:, i].sum() - matrix[i, i]
        tn = total - matrix[i, :].sum() - matrix[:, i].sum() + matrix[i, i]
        values.append(fp / (fp + tn) if fp + tn else 0.0)
    return float(np.mean(values))


def evaluate(x_train, y_train, x_test, y_test, experiment, output):
    rows = []
    with (output / f"{experiment}.results.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"Experiment: {experiment}\n")
        for estimator in models():
            estimator.fit(x_train, y_train)
            predicted = estimator.predict(x_test)
            row = {"experiment": experiment, "model": type(estimator).__name__, "accuracy": accuracy_score(y_test, predicted), "precision_macro": precision_score(y_test, predicted, average="macro", zero_division=0), "recall_macro": recall_score(y_test, predicted, average="macro", zero_division=0), "f1_macro": f1_score(y_test, predicted, average="macro", zero_division=0), "fpr_macro": macro_fpr(y_test, predicted)}
            rows.append(row)
            stream.write(f"\n{row['model']}\nAccuracy: {row['accuracy']*100:.4f}%\nPrecision (macro): {row['precision_macro']*100:.4f}%\nRecall (macro): {row['recall_macro']*100:.4f}%\nF1-score (macro): {row['f1_macro']*100:.4f}%\nFalse positive rate (macro): {row['fpr_macro']*100:.4f}%\n")
    return rows


def plot_distribution(labels, title, destination):
    counts = pd.Series(labels).value_counts().sort_index()
    names = [FILE_TYPES[int(i)] if 0 <= int(i) < len(FILE_TYPES) else str(i) for i in counts.index]
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(names, counts.values, color="#2563eb")
    ax.bar_label(bars, padding=2, fontsize=7)
    ax.set(title=title, xlabel="File type class", ylabel="Samples")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sift_512.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/generated"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots/generated"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}. See data/README.md.")
    frame = pd.read_csv(args.data)
    if TARGET not in frame.columns:
        parser.error(f"Missing target column '{TARGET}'.")
    columns = [name for name in frame.columns if name != TARGET]
    if len(columns) != N_FEATURES:
        parser.error(f"Expected 512 feature columns, found {len(columns)}.")
    if frame[columns].isna().any().any() or frame[TARGET].isna().any():
        parser.error("Dataset contains missing values.")
    x, y = frame[columns], frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_distribution(y_train, "Training split before SMOTE", args.plots_dir / "class_distribution_before_smote.png")
    rows = evaluate(x_train, y_train, x_test, y_test, "baseline", args.results_dir)
    x_balanced, y_balanced = SMOTE(random_state=SEED).fit_resample(x_train, y_train)
    plot_distribution(y_balanced, "Training split after SMOTE", args.plots_dir / "class_distribution_after_smote.png")
    rows += evaluate(x_balanced, y_balanced, x_test, y_test, "smote", args.results_dir)
    scaler = MinMaxScaler()
    x_balanced_scaled = scaler.fit_transform(x_balanced)
    x_test_scaled = scaler.transform(x_test)
    selector = SelectKBest(chi2, k=min(SELECTED, x_balanced_scaled.shape[1]))
    x_train_selected = selector.fit_transform(x_balanced_scaled, y_balanced)
    x_test_selected = selector.transform(x_test_scaled)
    rows += evaluate(x_train_selected, y_balanced, x_test_selected, y_test, "feature_selection", args.results_dir)
    pd.DataFrame(rows).to_csv(args.results_dir / "metrics.csv", index=False)
    print(f"Metrics written to {args.results_dir / 'metrics.csv'}")
    print("This script does not export a fitted model or raw-file inference pipeline.")

if __name__ == "__main__":
    main()

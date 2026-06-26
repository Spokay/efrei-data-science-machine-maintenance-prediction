import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelBinarizer

from config import CLASS_ORDER, CV_FOLDS, FIGURES_DIR, METRICS_DIR, RANDOM_STATE


def _ensure_dirs():
    FIGURES_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true, y_pred, y_proba, label: str) -> dict:
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    y_proba_arr = np.array(y_proba)

    roc_auc = roc_auc_score(y_bin, y_proba_arr, multi_class="ovr", average="macro")
    pr_auc = average_precision_score(y_bin, y_proba_arr, average="macro")

    return {
        "model": label,
        "recall_macro":  round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":      round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "roc_auc_macro": round(roc_auc, 4),
        "pr_auc_macro":  round(pr_auc, 4),
    }


def plot_confusion_matrix(y_true, y_pred, label: str):
    _ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=CLASS_ORDER,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix {label}")
    plt.tight_layout()
    path = FIGURES_DIR / f"confusion_matrix_{label.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_roc_curves(y_true, y_proba, classes, label: str):
    _ensure_dirs()
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    y_proba_arr = np.array(y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_arr[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba_arr[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curves {label}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = FIGURES_DIR / f"roc_{label.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def cross_validate_model(pipeline, X_train, y_train, label: str) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=["recall_macro", "f1_macro", "roc_auc_ovr_weighted"],
        n_jobs=-1,
    )
    result = {
        "model":          label,
        "cv_recall_mean": round(scores["test_recall_macro"].mean(), 4),
        "cv_recall_std":  round(scores["test_recall_macro"].std(), 4),
        "cv_f1_mean":     round(scores["test_f1_macro"].mean(), 4),
        "cv_f1_std":      round(scores["test_f1_macro"].std(), 4),
        "cv_auc_mean":    round(scores["test_roc_auc_ovr_weighted"].mean(), 4),
        "cv_auc_std":     round(scores["test_roc_auc_ovr_weighted"].std(), 4),
    }
    print(
        f"CV Recall={result['cv_recall_mean']}±{result['cv_recall_std']}  "
        f"F1={result['cv_f1_mean']}±{result['cv_f1_std']}  "
        f"AUC={result['cv_auc_mean']}±{result['cv_auc_std']}"
    )
    return result


def plot_feature_importance(importances: np.ndarray, feature_names: list, label: str):
    _ensure_dirs()
    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
    )
    ax.set_title(f"Feature Importance {label}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = FIGURES_DIR / f"feature_importance_{label.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_shap_summary(model, X_transformed: np.ndarray, feature_names: list, label: str):
    _ensure_dirs()
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
    except Exception:
        background = shap.kmeans(X_transformed, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_transformed[:200], nsamples=100)

    # Reduce to (n_features,): mean |SHAP| across samples and classes.
    # SHAP < 0.42 returns list of (n_samples, n_features) per class.
    # SHAP >= 0.42 returns (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features).
    sv = np.array(shap_values)
    if sv.ndim == 3 and sv.shape[0] < sv.shape[1]:
        mean_abs_shap = np.abs(sv).mean(axis=(0, 1))   # (n_classes, n_samples, n_features)
    elif sv.ndim == 3:
        mean_abs_shap = np.abs(sv).mean(axis=(0, 2))   # (n_samples, n_features, n_classes)
    else:
        mean_abs_shap = np.abs(sv).mean(axis=0)        # (n_samples, n_features)

    feature_names_arr = np.array(list(feature_names))
    idx = np.argsort(mean_abs_shap)[::-1][:20]
    idx_asc = idx[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        list(feature_names_arr[idx_asc]),
        mean_abs_shap[idx_asc].tolist(),
    )
    ax.set_title(f"SHAP Mean |value| {label}")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    path = FIGURES_DIR / f"shap_{label.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def save_comparison_table(rows: list[dict]):
    _ensure_dirs()
    df = pd.DataFrame(rows)
    path = METRICS_DIR / "comparison_table.csv"
    df.to_csv(path, index=False)
    print(f"\nComparison table saved → {path}")
    print(df.to_string(index=False))

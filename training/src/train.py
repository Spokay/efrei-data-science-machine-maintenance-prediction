"""
Training pipeline — Industrial Predictive Maintenance (failure_type classification).

Usage:
    python src/train.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from config import API_MODELS_DIR, CV_FOLDS, FEATURES_CAT, FEATURES_NUM, FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_STATE
from evaluation import (
    compute_metrics,
    cross_validate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    plot_shap_summary,
    save_comparison_table,
)
from models import (
    build_tf_mlp,
    compute_tf_class_weights,
    get_logistic_regression,
    get_random_forest,
    get_xgboost,
)
from preprocessing import build_preprocessor, load_data, split_data


def _get_feature_names(fitted_preprocessor) -> list[str]:
    cat_names = (
        fitted_preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(FEATURES_CAT)
        .tolist()
    )
    return FEATURES_NUM + cat_names


def _export_onnx(model: tf.keras.Model):
    savedmodel_path = str(MODELS_DIR / "tf_savedmodel")
    model.export(savedmodel_path)
    result = subprocess.run(
        [
            "python", "-m", "tf2onnx.convert",
            "--saved-model", savedmodel_path,
            "--output", str(MODELS_DIR / "model.onnx"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"ONNX model saved : {MODELS_DIR / 'model.onnx'}")
    else:
        print(f"ONNX export failed:\n{result.stderr[-500:]}")


def train_sklearn_model(pipeline, X_train, y_train, X_test, y_test,
                        label: str, sample_weight=None) -> dict:
    print(f"\n{'='*50}")
    print(f"Training: {label}")

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight

    pipeline.fit(X_train, y_train, **fit_kwargs)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = compute_metrics(y_test, y_pred, y_proba, label)
    plot_confusion_matrix(y_test, y_pred, label)
    plot_roc_curves(y_test, y_proba, pipeline.classes_, label)
    plot_pr_curves(y_test, y_proba, pipeline.classes_, label)

    path = MODELS_DIR / f"{label}.pkl"
    joblib.dump(pipeline, path)
    print(f"Model saved to : {path}")

    return metrics


def train_xgboost_model(X_train, y_train, X_test, y_test,
                        label_encoder: LabelEncoder, feature_names: list) -> dict:
    print(f"\n{'='*50}")
    print("Training: XGBoost")

    y_train_enc = label_encoder.transform(y_train)

    class_weights_arr = compute_class_weight(
        "balanced", classes=label_encoder.classes_, y=y_train
    )
    sample_weight = np.array([class_weights_arr[c] for c in y_train_enc])

    pipeline = get_xgboost(n_classes=len(label_encoder.classes_))
    pipeline.fit(X_train, y_train_enc, clf__sample_weight=sample_weight)

    y_pred_enc = pipeline.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    y_proba = pipeline.predict_proba(X_test)

    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = compute_metrics(y_test, y_pred, y_proba, "XGBoost")
    plot_confusion_matrix(y_test, y_pred, "XGBoost")
    plot_roc_curves(y_test, y_proba, label_encoder.classes_, "XGBoost")
    plot_pr_curves(y_test, y_proba, label_encoder.classes_, "XGBoost")

    xgb_clf = pipeline.named_steps["clf"]
    plot_feature_importance(xgb_clf.feature_importances_, feature_names, "XGBoost")

    joblib.dump(pipeline, MODELS_DIR / "XGBoost.pkl")
    print(f"Model saved to : {MODELS_DIR / 'XGBoost.pkl'}")

    return metrics


def train_tf_model(X_train_t: np.ndarray, y_train_enc: np.ndarray,
                   X_test_t: np.ndarray, y_test_enc: np.ndarray,
                   label_encoder: LabelEncoder) -> dict:
    print(f"\n{'='*50}")
    print("Training: TensorFlow MLP")

    n_classes = len(label_encoder.classes_)
    class_weights = compute_tf_class_weights(y_train_enc)
    model = build_tf_mlp(X_train_t.shape[1], n_classes)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
    tr_idx, val_idx = next(sss.split(X_train_t, y_train_enc))

    model.fit(
        X_train_t[tr_idx], y_train_enc[tr_idx],
        epochs=50,
        batch_size=256,
        validation_data=(X_train_t[val_idx], y_train_enc[val_idx]),
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )],
        verbose=1,
    )

    y_proba = model.predict(X_test_t)
    y_pred = label_encoder.inverse_transform(np.argmax(y_proba, axis=1))
    y_test = label_encoder.inverse_transform(y_test_enc)

    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = compute_metrics(y_test, y_pred, y_proba, "TF_MLP")
    plot_confusion_matrix(y_test, y_pred, "TF_MLP")
    plot_roc_curves(y_test, y_proba, label_encoder.classes_, "TF_MLP")
    plot_pr_curves(y_test, y_proba, label_encoder.classes_, "TF_MLP")

    model.save(str(MODELS_DIR / "tf_model.keras"))
    print(f"TF model saved → {MODELS_DIR / 'tf_model.keras'}")

    _export_onnx(model)

    return metrics


def cross_validate_tf_model(X_train_t: np.ndarray, y_train_enc: np.ndarray,
                             label_encoder: LabelEncoder) -> dict:
    print(f"\n{'='*50}")
    print("CV: TensorFlow MLP")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_features, n_classes = X_train_t.shape[1], len(label_encoder.classes_)
    lb = LabelBinarizer()
    lb.fit(label_encoder.classes_)
    recalls, f1s, aucs = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_t, y_train_enc), 1):
        X_fold_tr, X_fold_val = X_train_t[tr_idx], X_train_t[val_idx]
        y_fold_tr, y_fold_val = y_train_enc[tr_idx], y_train_enc[val_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
        es_tr, es_val = next(sss.split(X_fold_tr, y_fold_tr))

        model = build_tf_mlp(n_features, n_classes)
        cw = compute_tf_class_weights(y_fold_tr[es_tr])
        model.fit(
            X_fold_tr[es_tr], y_fold_tr[es_tr],
            epochs=50, batch_size=256, verbose=0,
            class_weight=cw,
            validation_data=(X_fold_tr[es_val], y_fold_tr[es_val]),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )],
        )

        y_proba = model.predict(X_fold_val, verbose=0)
        y_true_str = label_encoder.inverse_transform(y_fold_val)
        y_pred_str = label_encoder.inverse_transform(np.argmax(y_proba, axis=1))

        recalls.append(recall_score(y_true_str, y_pred_str, average="macro", zero_division=0))
        f1s.append(f1_score(y_true_str, y_pred_str, average="macro", zero_division=0))
        aucs.append(roc_auc_score(lb.transform(y_true_str), y_proba, multi_class="ovr", average="macro"))
        print(f"  Fold {fold}: Recall={recalls[-1]:.4f}  F1={f1s[-1]:.4f}  AUC={aucs[-1]:.4f}")

    result = {
        "model": "TF_MLP",
        "cv_recall_mean": round(np.mean(recalls), 4),
        "cv_recall_std":  round(np.std(recalls), 4),
        "cv_f1_mean":     round(np.mean(f1s), 4),
        "cv_f1_std":      round(np.std(f1s), 4),
        "cv_auc_mean":    round(np.mean(aucs), 4),
        "cv_auc_std":     round(np.std(aucs), 4),
    }
    print(f"CV Recall={result['cv_recall_mean']}±{result['cv_recall_std']}  "
          f"F1={result['cv_f1_mean']}±{result['cv_f1_std']}  "
          f"AUC={result['cv_auc_mean']}±{result['cv_auc_std']}")
    return result


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Class distribution (train):\n{y_train.value_counts()}\n")

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    all_metrics = []
    cv_metrics = []

    # 1. Logistic Regression (baseline)
    lr = get_logistic_regression()
    cv_metrics.append(cross_validate_model(lr, X_train, y_train, "Logistic_Regression"))
    all_metrics.append(train_sklearn_model(lr, X_train, y_train, X_test, y_test, "Logistic_Regression"))

    # 2. Random Forest
    rf = get_random_forest()
    cv_metrics.append(cross_validate_model(rf, X_train, y_train, "Random_Forest"))
    all_metrics.append(train_sklearn_model(rf, X_train, y_train, X_test, y_test, "Random_Forest"))

    feature_names = _get_feature_names(rf.named_steps["prep"])
    plot_feature_importance(rf.named_steps["clf"].feature_importances_, feature_names, "Random_Forest")

    # 3. XGBoost — CV sur labels encodés (entiers), cohérent avec l'entraînement
    cv_metrics.append(cross_validate_model(get_xgboost(n_classes=len(le.classes_)), X_train, y_train_enc, "XGBoost"))
    all_metrics.append(train_xgboost_model(X_train, y_train, X_test, y_test, le, feature_names))

    # 4. TensorFlow MLP — CV stratifié puis entraînement final
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    cv_metrics.append(cross_validate_tf_model(X_train_t, y_train_enc, le))
    all_metrics.append(train_tf_model(X_train_t, y_train_enc, X_test_t, y_test_enc, le))

    # SHAP on Random Forest
    print("\nComputing SHAP for Random Forest...")
    X_test_t_rf = rf.named_steps["prep"].transform(X_test)
    plot_shap_summary(rf.named_steps["clf"], X_test_t_rf, feature_names, "Random_Forest")

    # Save results
    save_comparison_table(all_metrics)

    cv_df = pd.DataFrame(cv_metrics)
    cv_df.to_csv(METRICS_DIR / "cv_results.csv", index=False)
    print(f"CV results saved → {METRICS_DIR / 'cv_results.csv'}")

    best = max(all_metrics, key=lambda x: x["recall_macro"])
    print(f"\nBest model by Recall macro: {best['model']}  "
          f"Recall={best['recall_macro']}  F1={best['f1_macro']}")

    API_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = API_MODELS_DIR / "XGBoost.pkl"
    shutil.copy2(MODELS_DIR / "XGBoost.pkl", dest)
    print(f"Model copied to API → {dest}")


if __name__ == "__main__":
    main()

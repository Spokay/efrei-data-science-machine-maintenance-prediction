import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from preprocessing import build_preprocessor


def get_logistic_regression() -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])


def get_random_forest() -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def get_xgboost(n_classes: int) -> Pipeline:
    # XGBoost multiclass requires integer-encoded labels and explicit num_class.
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


def get_smote_pipeline(base_pipeline: Pipeline) -> ImbPipeline:
    prep_step, clf_step = base_pipeline.steps[0], base_pipeline.steps[-1]
    return ImbPipeline([prep_step, ("smote", SMOTE(random_state=42)), clf_step])


def get_undersample_pipeline(base_pipeline: Pipeline) -> ImbPipeline:
    prep_step, clf_step = base_pipeline.steps[0], base_pipeline.steps[-1]
    return ImbPipeline([prep_step, ("under", RandomUnderSampler(random_state=42)), clf_step])


def build_tf_mlp(n_features: int, n_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_tf_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(enumerate(weights))

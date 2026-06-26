import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    COLS_TO_DROP,
    DATA_PATH,
    FEATURES_CAT,
    FEATURES_NUM,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from sklearn.model_selection import train_test_split


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=COLS_TO_DROP)
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, FEATURES_NUM),
        ("cat", categorical_pipeline, FEATURES_CAT),
    ])

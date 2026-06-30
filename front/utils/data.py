"""Chargement (mis en cache) des données d'entraînement et des métriques offline.

Ces fichiers sont indépendants de l'API : ils permettent d'explorer les données
et de comparer les modèles même si l'API n'est pas démarrée.
"""

import pandas as pd
import streamlit as st

from .config import DATA_PATH, METRICS_DIR


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


@st.cache_data
def load_comparison_table() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "comparison_table.csv")


@st.cache_data
def load_cv_results() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "cv_results.csv")

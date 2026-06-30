import json

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.api_client import APIError, get_model_info, predict
from utils.config import (
    DEFAULT_MACHINE_TYPES,
    DEFAULT_OPERATING_MODES,
    RAW_FEATURES,
    TEST_VALUES_PATH,
)
from utils.ui import render_api_sidebar, render_prediction_result

st.set_page_config(page_title="Prédiction", page_icon="🔮", layout="wide")
render_api_sidebar()

st.title("🔮 Simulateur de prédiction")
st.caption(
    "Saisissez un scénario capteur (ou chargez un lot) pour obtenir, en direct via l'API, "
    "le type de défaillance le plus probable et une recommandation d'action."
)

try:
    model_info = get_model_info()
    machine_types = model_info["categorical_values"]["machine_type"]
    operating_modes = model_info["categorical_values"]["operating_mode"]
except APIError:
    machine_types = DEFAULT_MACHINE_TYPES
    operating_modes = DEFAULT_OPERATING_MODES
    st.info("API inaccessible : valeurs catégorielles par défaut utilisées (la prédiction nécessitera l'API).")

tab_manual, tab_batch = st.tabs(["🧪 Scénario manuel", "📂 Lot (CSV)"])

# --- Scénario manuel -------------------------------------------------------
with tab_manual:
    examples = {}
    if TEST_VALUES_PATH.exists():
        examples = json.loads(TEST_VALUES_PATH.read_text(encoding="utf-8"))

    preset = st.selectbox("Charger un exemple type (optionnel)", ["—"] + list(examples.keys()))
    defaults = examples.get(preset, {}) if preset != "—" else {}

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        vibration_rms = c1.number_input(
            "Vibration RMS", min_value=0.0, value=float(defaults.get("vibration_rms", 1.5)), step=0.01
        )
        temperature_motor = c1.number_input(
            "Température moteur (°C)", value=float(defaults.get("temperature_motor", 55.0)), step=0.1
        )
        current_phase_avg = c1.number_input(
            "Courant de phase moyen (A)", min_value=0.0, value=float(defaults.get("current_phase_avg", 8.0)), step=0.1
        )
        pressure_level = c2.number_input(
            "Niveau de pression", min_value=0.0, value=float(defaults.get("pressure_level", 40.0)), step=0.1
        )
        rpm = c2.number_input("Vitesse (RPM)", min_value=0.0, value=float(defaults.get("rpm", 1500.0)), step=10.0)
        hours_since_maintenance = c2.number_input(
            "Heures depuis dernière maintenance",
            min_value=0.0,
            value=float(defaults.get("hours_since_maintenance", 150.0)),
            step=1.0,
        )
        ambient_temp = c3.number_input(
            "Température ambiante (°C)", value=float(defaults.get("ambient_temp", 14.0)), step=0.1
        )
        machine_type = c3.selectbox(
            "Type de machine",
            machine_types,
            index=machine_types.index(defaults["machine_type"]) if defaults.get("machine_type") in machine_types else 0,
        )
        operating_mode = c3.selectbox(
            "Mode opératoire",
            operating_modes,
            index=operating_modes.index(defaults["operating_mode"])
            if defaults.get("operating_mode") in operating_modes
            else 0,
        )

        submitted = st.form_submit_button("Lancer la prédiction", type="primary")

    if submitted:
        payload = {
            "vibration_rms": vibration_rms,
            "temperature_motor": temperature_motor,
            "current_phase_avg": current_phase_avg,
            "pressure_level": pressure_level,
            "rpm": rpm,
            "hours_since_maintenance": hours_since_maintenance,
            "ambient_temp": ambient_temp,
            "machine_type": machine_type,
            "operating_mode": operating_mode,
        }
        try:
            result = predict(payload)
        except APIError as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")
        else:
            render_prediction_result(result)

# --- Lot (CSV) --------------------------------------------------------------
with tab_batch:
    st.markdown(f"Chargez un CSV contenant les 9 colonnes brutes : `{', '.join(RAW_FEATURES)}`.")
    file = st.file_uploader("Fichier CSV", type="csv")

    if file is not None:
        batch_df = pd.read_csv(file)
        missing_cols = set(RAW_FEATURES) - set(batch_df.columns)
        if missing_cols:
            st.error(f"Colonnes manquantes : {', '.join(sorted(missing_cols))}")
        elif st.button("Lancer les prédictions par lot", type="primary"):
            progress = st.progress(0.0)
            predicted_classes, is_failures, probabilities = [], [], []
            errors = []
            n = len(batch_df)
            for i, (_, row) in enumerate(batch_df.iterrows()):
                try:
                    res = predict(row[RAW_FEATURES].to_dict())
                    predicted_classes.append(res["predicted_class"])
                    is_failures.append(res["is_failure"])
                    probabilities.append(res["probability"])
                except APIError as e:
                    predicted_classes.append(None)
                    is_failures.append(None)
                    probabilities.append(None)
                    errors.append(f"Ligne {i} : {e}")
                progress.progress((i + 1) / n)

            out = batch_df.copy()
            out["predicted_class"] = predicted_classes
            out["is_failure"] = is_failures
            out["probability"] = probabilities
            st.session_state["batch_results"] = out
            if errors:
                st.warning(f"{len(errors)} ligne(s) en erreur (voir détail ci-dessous).")
                with st.expander("Détail des erreurs"):
                    st.write(errors)

    if "batch_results" in st.session_state:
        out = st.session_state["batch_results"]
        st.subheader("Résultats")

        c1, c2, c3 = st.columns(3)
        c1.metric("Lignes traitées", len(out))
        fail_rate = out["is_failure"].mean()
        c2.metric("Taux de défaillance prédit", f"{fail_rate:.1%}" if pd.notna(fail_rate) else "—")
        top_class = out["predicted_class"].value_counts().idxmax() if out["predicted_class"].notna().any() else "—"
        c3.metric("Classe la plus fréquente", top_class)

        fig = px.histogram(out, x="predicted_class", color="predicted_class")
        st.plotly_chart(fig, width="stretch")

        st.dataframe(out, width="stretch")
        st.download_button(
            "Télécharger les résultats (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv",
        )

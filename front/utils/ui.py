"""Composants UI partagés entre les pages du dashboard."""

import pandas as pd
import plotly.express as px
import streamlit as st

from .api_client import APIError, check_health
from .config import DEFAULT_API_URL, FAILURE_ADVICE


def render_api_sidebar() -> dict | None:
    """Affiche le statut de connexion à l'API dans la sidebar et renvoie /health (ou None)."""
    st.sidebar.markdown("## 🔌 Connexion API")
    current = st.session_state.get("api_url", DEFAULT_API_URL)
    url = st.sidebar.text_input(
        "URL de l'API",
        value=current,
        help="Endpoint FastAPI exposant /health, /predict et /model-info.",
    )
    st.session_state["api_url"] = url.strip() or DEFAULT_API_URL

    try:
        health = check_health()
    except APIError as e:
        st.sidebar.error("❌ API inaccessible")
        st.sidebar.caption(str(e))
        return None

    if health.get("model_loaded"):
        st.sidebar.success("✅ API connectée — modèle chargé")
    else:
        st.sidebar.warning("⚠️ API connectée — mode dégradé (pas de modèle)")
    st.sidebar.caption(f"Modèle : {health.get('model_path', '—')}")
    return health


def render_prediction_result(result: dict) -> None:
    """Affiche un résultat de /predict de façon orientée décision (alerte + recommandation)."""
    predicted_class = result["predicted_class"]
    advice = FAILURE_ADVICE.get(predicted_class, {})
    label = advice.get("label", predicted_class)

    if result["is_failure"]:
        st.error(f"⚠️ Défaillance prédite : **{label}** (probabilité {result['probability']:.1%})")
    else:
        st.success(f"✅ Aucune défaillance détectée (probabilité {result['probability']:.1%})")

    if advice.get("action"):
        st.markdown(f"**Action recommandée :** {advice['action']}")

    probs = pd.DataFrame(
        sorted(result["class_probabilities"].items(), key=lambda kv: -kv[1]),
        columns=["classe", "probabilité"],
    )
    fig = px.bar(probs, x="probabilité", y="classe", orientation="h", text_auto=".1%")
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(yaxis_title=None, xaxis_title="Probabilité")
    st.plotly_chart(fig, width="stretch")

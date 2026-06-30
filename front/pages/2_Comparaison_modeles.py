import plotly.express as px
import streamlit as st

from utils.api_client import APIError, get_model_info
from utils.config import FIGURES_DIR
from utils.data import load_comparison_table, load_cv_results
from utils.ui import render_api_sidebar

st.set_page_config(page_title="Comparaison des modèles", page_icon="🤖", layout="wide")
render_api_sidebar()

st.title("Comparaison des modèles")

comp = load_comparison_table()
best_row = comp.sort_values("recall_macro", ascending=False).iloc[0]
st.markdown(
    f"**Modèle retenu en production : `{best_row['model']}`** "
    f"(Recall macro = {best_row['recall_macro']:.4f} — métrique prioritaire compte tenu "
    "du déséquilibre des classes 31:1 ; un faux négatif coûte plus cher qu'une fausse alarme)."
)

melted = comp.melt(id_vars="model", var_name="metric", value_name="value")
fig = px.bar(melted, x="model", y="value", color="metric", barmode="group", text_auto=".3f")
fig.update_layout(yaxis_range=[0, 1])
st.plotly_chart(fig, width="stretch")

st.dataframe(
    comp.style.highlight_max(axis=0, subset=[c for c in comp.columns if c != "model"], color="#2e7d32"),
    width="stretch",
)

st.subheader("Validation croisée (5 folds stratifiés)")
cv = load_cv_results()
fig_cv = px.bar(cv, x="model", y="cv_recall_mean", error_y="cv_recall_std", text_auto=".3f")
fig_cv.update_yaxes(title="Recall macro (CV)", range=[0, 1])
st.plotly_chart(fig_cv, width="stretch")
st.dataframe(cv, width="stretch")

st.divider()
st.subheader("Modèle actuellement déployé sur l'API")
try:
    info = get_model_info()
except APIError as e:
    st.warning(f"Impossible de récupérer les informations du modèle déployé : {e}")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Type de pipeline", info.get("model_type") or "—")
    c2.metric("Features après preprocessing", info.get("n_features_after_preprocessing") or "—")
    c3.metric("Version API", info.get("version") or "—")
    with st.expander("Détail complet (/model-info)"):
        st.json(info)

st.divider()
st.subheader("Visualisations détaillées par modèle")
model_files = {
    "XGBoost": "XGBoost",
    "Random_Forest": "Random Forest",
    "Logistic_Regression": "Logistic Regression",
    "TF_MLP": "TF MLP",
}
selected = st.selectbox("Modèle", list(model_files.keys()), format_func=lambda k: model_files[k])

col_a, col_b = st.columns(2)
cm_path = FIGURES_DIR / f"confusion_matrix_{selected}.png"
roc_path = FIGURES_DIR / f"roc_{selected}.png"
if cm_path.exists():
    col_a.image(str(cm_path), caption=f"Matrice de confusion — {model_files[selected]}", width="stretch")
if roc_path.exists():
    col_b.image(str(roc_path), caption=f"Courbe ROC (optimiste sur classes déséquilibrées) — {model_files[selected]}", width="stretch")

col_pr, col_empty = st.columns(2)
pr_path = FIGURES_DIR / f"pr_{selected}.png"
if pr_path.exists():
    col_pr.image(str(pr_path), caption=f"Courbe Précision-Rappel (à privilégier, déséquilibre 31:1) — {model_files[selected]}", width="stretch")

col_c, col_d = st.columns(2)
fi_path = FIGURES_DIR / f"feature_importance_{selected}.png"
shap_path = FIGURES_DIR / f"shap_{selected}.png"
if fi_path.exists():
    col_c.image(str(fi_path), caption=f"Importance des features — {model_files[selected]}", width="stretch")
if shap_path.exists():
    col_d.image(str(shap_path), caption=f"SHAP — {model_files[selected]}", width="stretch")

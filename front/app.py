import plotly.express as px
import streamlit as st

from utils.data import load_raw_data
from utils.ui import render_api_sidebar

st.set_page_config(page_title="Maintenance Prédictive — Dashboard", page_icon="🛠️", layout="wide")

render_api_sidebar()

st.title("🛠️ Dashboard de Maintenance Prédictive")
st.caption(
    "Outil décisionnel autonome : indicateurs clés, exploration des données, "
    "comparaison des modèles et simulation de scénarios capteurs."
)

df = load_raw_data()

total = len(df)
failure_rate = (df["failure_type"] != "none").mean()
n_machines = df["machine_id"].nunique()
period = f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations", f"{total:,}")
col2.metric("Taux de défaillance global", f"{failure_rate:.1%}")
col3.metric("Machines suivies", n_machines)
col4.metric("Période couverte", period)

st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Répartition des types de défaillance")
    counts = df["failure_type"].value_counts().reset_index()
    counts.columns = ["failure_type", "count"]
    fig = px.bar(counts, x="failure_type", y="count", color="failure_type", text="count")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

with c2:
    st.subheader("Défaillances par type de machine")
    sub = df[df["failure_type"] != "none"]
    grouped = sub.groupby(["machine_type", "failure_type"]).size().reset_index(name="count")
    fig = px.bar(grouped, x="machine_type", y="count", color="failure_type", barmode="stack")
    st.plotly_chart(fig, width="stretch")

st.subheader("Évolution du taux de défaillance dans le temps")
daily = (
    df.assign(date=df["timestamp"].dt.date, is_failure=df["failure_type"] != "none")
    .groupby("date")["is_failure"]
    .mean()
    .reset_index(name="failure_rate")
)
fig = px.line(daily, x="date", y="failure_rate")
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")

st.divider()
st.markdown(
    """
### Navigation

- **📊 Exploration** : indicateurs clés, distributions, corrélations, filtres par machine et période.
- **🤖 Comparaison des modèles** : performances comparées (Recall, F1, ROC-AUC, PR-AUC), interprétabilité (SHAP).
- **🔮 Prédiction** : simuler un scénario capteur (manuel ou par lot) et obtenir une recommandation immédiate.

Utilisez le menu dans la barre latérale pour naviguer entre les pages.
"""
)

import plotly.express as px
import streamlit as st

from utils.config import NUMERIC_FEATURES
from utils.data import load_raw_data
from utils.ui import render_api_sidebar

st.set_page_config(page_title="Exploration des données", page_icon="📊", layout="wide")
render_api_sidebar()

st.title("📊 Exploration des données")

df = load_raw_data()

st.sidebar.markdown("## 🔍 Filtres")
machine_types = st.sidebar.multiselect(
    "Type de machine", sorted(df["machine_type"].unique()), default=sorted(df["machine_type"].unique())
)
operating_modes = st.sidebar.multiselect(
    "Mode opératoire", sorted(df["operating_mode"].unique()), default=sorted(df["operating_mode"].unique())
)
date_min, date_max = df["timestamp"].min().date(), df["timestamp"].max().date()
date_range = st.sidebar.date_input("Période", value=(date_min, date_max), min_value=date_min, max_value=date_max)

mask = df["machine_type"].isin(machine_types) & df["operating_mode"].isin(operating_modes)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    mask &= (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
fdf = df[mask]

st.caption(f"{len(fdf):,} observations sélectionnées sur {len(df):,}")

col1, col2, col3 = st.columns(3)
col1.metric("Observations filtrées", f"{len(fdf):,}")
col2.metric("Taux de défaillance", f"{(fdf['failure_type'] != 'none').mean():.1%}" if len(fdf) else "—")
col3.metric("Machines", fdf["machine_id"].nunique())

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Distribution cible", "Distributions des features", "Boxplots par classe", "Corrélations", "Valeurs manquantes"]
)

with tab1:
    counts = fdf["failure_type"].value_counts(normalize=True).reset_index()
    counts.columns = ["failure_type", "proportion"]
    fig = px.bar(counts, x="failure_type", y="proportion", color="failure_type", text_auto=".1%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

with tab2:
    feat = st.selectbox("Feature", NUMERIC_FEATURES, key="hist_feat")
    fig = px.histogram(fdf, x=feat, color="failure_type", marginal="box", nbins=50)
    st.plotly_chart(fig, width="stretch")

with tab3:
    feat2 = st.selectbox("Feature", NUMERIC_FEATURES, key="box_feat")
    fig = px.box(fdf, x="failure_type", y=feat2, color="failure_type")
    st.plotly_chart(fig, width="stretch")

with tab4:
    if len(fdf) > 1:
        corr = fdf[NUMERIC_FEATURES].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Pas assez de données pour calculer une corrélation.")

with tab5:
    missing = fdf[NUMERIC_FEATURES].isna().sum().reset_index()
    missing.columns = ["feature", "missing_count"]
    missing["missing_pct"] = missing["missing_count"] / len(fdf) * 100 if len(fdf) else 0
    fig = px.bar(missing, x="feature", y="missing_pct", text_auto=".1f")
    fig.update_yaxes(title="% de valeurs manquantes")
    st.plotly_chart(fig, width="stretch")

st.subheader("Aperçu des données filtrées")
st.dataframe(fdf.head(200), width="stretch")

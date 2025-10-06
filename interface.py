import os
import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.tauxcharge.elastic.elasticsearch import ElasticFetcher
from src.tauxcharge.preprocessing.custom_preprocessor import CustomPreprocessor
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="📈 Observations & Prédictions", layout="wide")
st.title("📊 Observations & Prédictions depuis Elasticsearch")

# ⏱️ Rafraîchissement automatique toutes les 5 minutes
st_autorefresh(interval=10 * 60 * 1000, key="datarefresh")

# Initialisation des données accumulées pour les observations
if "observations" not in st.session_state:
    df_obs = ElasticFetcher().fetch_last_days(day=1)
    pre = CustomPreprocessor()
    df_obs = pre.fit_transform(df_obs)
    st.session_state["observations"] = df_obs
    st.session_state["last_update"] = datetime.datetime.now()

# Fonction pour mettre à jour les observations
def update_observations():
    new_data = ElasticFetcher().fetch_last_hour()
    new_data = preprocessing_pipe.fit_transform(new_data)
    if not new_data.empty:
        st.session_state["observations"] = pd.concat(
            [st.session_state["observations"], new_data]
        ).drop_duplicates()
    st.session_state["last_update"] = datetime.datetime.now()
    return st.session_state["observations"]

# Fonction pour récupérer les prédictions
def get_predictions():
    return ElasticFetcher(index=os.getenv("ES_INDEX_dest")).fetch_all()

# 🔄 Vérifier si une heure est passée depuis la dernière mise à jour
now = datetime.datetime.now()
elapsed = now - st.session_state.get("last_update", now)
if elapsed.total_seconds() >= 300:  # 1h
    with st.spinner("📥 Mise à jour automatique des observations…"):
        update_observations()
        st.success(f"✅ Observations mises à jour à {now.strftime('%H:%M:%S')}")

df_obs = st.session_state["observations"]
df_pred = get_predictions()

# --- Affichage des données brutes ---
st.markdown(f"### 📂 Observations (accumulées) — Dernière mise à jour : {st.session_state['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
if df_obs.empty:
    st.info("Aucune donnée pour les observations.")
else:
    st.dataframe(df_obs)

st.markdown("---")
st.markdown(f"### 📂 Prédictions (dernière récupération : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
if df_pred.empty:
    st.info("Aucune donnée pour les prédictions.")
else:
    st.dataframe(df_pred)

st.markdown("---")

# --- Sélection du name_id ---
if df_obs.empty or df_pred.empty:
    st.warning("Pas assez de données pour tracer les courbes.")
    st.stop()

# Extraire la liste des name_id présents dans les observations
name_ids = sorted(df_obs["name_id"].unique())
# 1️⃣ Extraire la liste des `names` uniques depuis df_obs
namess = sorted(df_obs["names"].unique())

# 2️⃣ Laisser l’utilisateur choisir un name
selected_name = st.selectbox("🔷 Sélectionnez un équipement :", namess)

# 3️⃣ Retrouver le name_id correspondant
name_id_selected = df_obs.loc[df_obs["names"] == selected_name, "name_id"].iloc[0]

# 4️⃣ Filtrer df_obs et df_pred avec le `name_id`
df_obs_id = df_obs[df_obs["name_id"] == name_id_selected].sort_values("date_mesure")
df_pred_id = df_pred[df_pred["name_id"] == name_id_selected].sort_values("predicted_for")

if df_obs_id.empty and df_pred_id.empty:
    st.info("Aucune donnée pour ce name_id.")
else:
    fig = go.Figure()

    if not df_obs_id.empty:
        fig.add_trace(go.Scatter(
            x=df_obs_id["date_mesure"],
            y=df_obs_id["taux_de_charge"],
            mode="lines+markers",
            name="Observations",
            line=dict(color="blue")
        ))

    if not df_pred_id.empty:
        fig.add_trace(go.Scatter(
            x=df_pred_id["predicted_for"],
            y=df_pred_id["predicted_taux_de_charge"],
            mode="lines+markers",
            name="Prédictions",
            line=dict(color="red", dash="dash")
        ))

    fig.update_layout(
        title=f"Taux de charge — name={selected_name}",
        xaxis_title="Temps",
        yaxis_title="Taux de charge",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


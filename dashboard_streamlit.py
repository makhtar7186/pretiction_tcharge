# dashboard_streamlit.py
# ----------------------------------------------------------
# Streamlit dashboard: OLT Load - Real vs Predicted
# (affiche la courbe complète des prédictions + fetch robuste)
# ----------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st

# Visuals
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Project imports (adapter le chemin à ton projet)
try:
    from src.tauxcharge.elastic.elasticsearch import ElasticFetcher
    from src.tauxcharge.preprocessing.custom_preprocessor import CustomPreprocessor
except Exception as e:
    ElasticFetcher = None
    CustomPreprocessor = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# ----------------------
# Streamlit page config
# ----------------------
st.set_page_config(
    page_title="OLT - Taux de charge | Réel vs Prédit",
    page_icon="📈",
    layout="wide",
)

st.title("📈 OLT – Taux de charge : Réel vs Prédit")
st.caption("Interface interactive pour comparer les données réelles et les prédictions (Streamlit + Plotly).")

# ----------------------
# Sidebar – parameters
# ----------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    day_range = st.slider("Fenêtre (jours) à récupérer", min_value=1, max_value=120, value=13, step=1)
    resample_rule = st.selectbox("Agrégation temporelle", options=["None (5min natif)", "15min", "30min", "60min"], index=0)
    show_altair = st.toggle("Activer un tracé Altair supplémentaire", value=False)
    show_full_pred = st.toggle(
        "Afficher toute la courbe des prédictions",
        value=True,
        help="Trace toutes les valeurs prédites du nom sélectionné, même sans valeur réelle correspondante."
    )
    st.markdown("---")
    st.write("Index ES prédictions (lecture) :")
    st.code(os.getenv("ES_INDEX_dest", "⚠️ ES_INDEX_dest non défini"), language="bash")
    auto_refresh = st.checkbox("Auto-refresh (1h)", value=False)
    # 👉 Mets 24h si tu préfères: st_autorefresh(interval=24*60*60*1000, ...)

if auto_refresh:
    st_autorefresh(interval=60 * 60 * 1000, limit=None, key="auto_refresh")

# ----------------------
# Helpers
# ----------------------
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]; y_pred = y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def compute_kpis(df_merge_name: pd.DataFrame) -> dict:
    if df_merge_name.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan}
    err = df_merge_name["taux_de_charge"] - df_merge_name["predicted_taux_de_charge"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = _safe_mape(df_merge_name["taux_de_charge"].values, df_merge_name["predicted_taux_de_charge"].values)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

def _try_fetch_pred_full(elt_save, day: int) -> pd.DataFrame:
    """
    Essaie plusieurs méthodes possibles pour récupérer les prédictions
    SANS tri ES sur .keyword, puis trie localement en pandas.
    """
    last_err = None
    for mname, kwargs in [
        ("fetch_pred", {"day": day}),
        ("fetch_last_days", {"day": day, "time_field": "predicted_for"}),  # si supporté par ta classe
        ("fetch_all", {}),  # dump complet
    ]:
        if hasattr(elt_save, mname):
            try:
                df = getattr(elt_save, mname)(**kwargs)
                if isinstance(df, pd.DataFrame):
                    return df
            except Exception as e:
                last_err = e
                continue
    if last_err is not None:
        raise last_err
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=True)
def load_real_and_pred(day: int) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """
    Récupère les données réelles (prétraitées) + TOUTES les prédictions,
    sans imposer de tri ES incompatible.
    """
    if _IMPORT_ERROR is not None:
        return pd.DataFrame(), pd.DataFrame(), f"Erreur d'import: {str(_IMPORT_ERROR)}"

    try:
        fetcher = ElasticFetcher()  # index par défaut (réel)
        pred_index = os.getenv("ES_INDEX_dest")
        if not pred_index:
            return pd.DataFrame(), pd.DataFrame(), "La variable d'environnement ES_INDEX_dest n'est pas définie."

        elt_save = ElasticFetcher(index=pred_index)
        pre = CustomPreprocessor()

        # --- Réel ---
        df_all = fetcher.fetch_last_days(day=day)
        df_all = pre.fit_transform(df_all)

        # --- Prédictions (robuste) ---
        #df_pred = _try_fetch_pred_full(elt_save, day=day)
        df_pred = elt_save.fetch_all()

        # Normalisation colonnes
        if "names" not in df_all.columns and "name" in df_all.columns:
            df_all = df_all.rename(columns={"name": "names"})

        # Datetime
        if "date_mesure" in df_all.columns:
            df_all["date_mesure"] = pd.to_datetime(df_all["date_mesure"], errors="coerce")

        # Certaines implémentations réutilisent "date_mesure" pour la date des prédictions
        if "predicted_for" not in df_pred.columns and "date_mesure" in df_pred.columns:
            df_pred = df_pred.rename(columns={"date_mesure": "predicted_for"})

        if "predicted_for" in df_pred.columns:
            df_pred["predicted_for"] = pd.to_datetime(df_pred["predicted_for"], errors="coerce")

        # Numeric
        if "taux_de_charge" in df_all.columns:
            df_all["taux_de_charge"] = pd.to_numeric(df_all["taux_de_charge"], errors="coerce")
        if "predicted_taux_de_charge" in df_pred.columns:
            df_pred["predicted_taux_de_charge"] = pd.to_numeric(df_pred["predicted_taux_de_charge"], errors="coerce")

        # name / names
        if "name" in df_pred.columns:
            df_pred["name"] = df_pred["name"].astype(str)
        elif "names" in df_pred.columns:
            df_pred = df_pred.rename(columns={"names": "name"})

        # Drop NaN critiques
        df_all = df_all.dropna(subset=["names", "date_mesure", "taux_de_charge"])
        df_pred = df_pred.dropna(subset=["name", "predicted_for", "predicted_taux_de_charge"])

        # Tri LOCAL (pas de tri ES)
        df_all = df_all.sort_values("date_mesure")
        df_pred = df_pred.sort_values("predicted_for")

        return df_all, df_pred, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"Erreur lors du chargement des données: {e}"

@st.cache_data(ttl=300, show_spinner=False)
def resample_if_needed(df_all: pd.DataFrame, df_pred: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optionnel : rééchantillonne les deux séries à la même fréquence (moyenne).
    """
    if rule == "None (5min natif)":
        return df_all.copy(), df_pred.copy()

    g_all = (
        df_all
        .set_index("date_mesure")
        .groupby("names")
        .resample(rule)["taux_de_charge"]
        .mean()
        .reset_index()
    )

    g_pred = (
        df_pred
        .set_index("predicted_for")
        .groupby("name")
        .resample(rule)["predicted_taux_de_charge"]
        .mean()
        .reset_index()
    )
    return g_all, g_pred

def merge_real_pred(df_all: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Merge exact sur (equipement, timestamp).
    Si tes prédictions ont un horizon décalé, gère le shift AVANT le merge.
    """
    df_merge = pd.merge(
        df_all.rename(columns={"names": "name", "date_mesure": "timestamp"}),
        df_pred.rename(columns={"predicted_for": "timestamp"}),
        on=["name", "timestamp"],
        how="inner",
    )
    df_merge = df_merge.rename(columns={"name": "names", "timestamp": "date_mesure"})
    return df_merge

# ----------------------
# Load data
# ----------------------
df_all, df_pred, load_err = load_real_and_pred(day=day_range)
if load_err:
    st.error(load_err)
    st.stop()

# Optional resampling
df_all_r, df_pred_r = resample_if_needed(df_all, df_pred, resample_rule)

# Equipment selection
all_names = sorted(pd.unique(df_all_r["names"].astype(str)))
default_name = all_names[0] if all_names else None
name_selected = st.selectbox("Choisir l'équipement (name)", options=all_names, index=0 if default_name else None)

if not name_selected:
    st.warning("Aucun équipement disponible.")
    st.stop()

# Sous-ensembles par équipement
df_real_sel = df_all_r[df_all_r["names"] == name_selected].sort_values("date_mesure")
df_pred_sel = df_pred_r[df_pred_r["name"] == name_selected].sort_values("predicted_for")

# Merge pour KPI/erreurs (zone de recouvrement)
df_merge_all = merge_real_pred(df_all_r, df_pred_r)
df_overlap = df_merge_all[df_merge_all["names"] == name_selected].sort_values("date_mesure")

# ----------------------
# KPI Cards (sur recouvrement)
# ----------------------
kpis = compute_kpis(df_overlap)
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{kpis['MAE']:.4f}" if np.isfinite(kpis['MAE']) else "—")
c2.metric("RMSE", f"{kpis['RMSE']:.4f}" if np.isfinite(kpis['RMSE']) else "—")
c3.metric("MAPE (%)", f"{kpis['MAPE (%)']:.2f}%" if np.isfinite(kpis['MAPE (%)']) else "—")

# ----------------------
# Comparative line chart
# ----------------------
fig = go.Figure()

# Réel
if not df_real_sel.empty:
    fig.add_trace(go.Scatter(
        x=df_real_sel["date_mesure"], y=df_real_sel["taux_de_charge"],
        mode="lines", name="Réel (taux_de_charge)"
    ))

# Prédit
if show_full_pred:
    # toute la courbe
    if not df_pred_sel.empty:
        fig.add_trace(go.Scatter(
            x=df_pred_sel["predicted_for"], y=df_pred_sel["predicted_taux_de_charge"],
            mode="lines", name="Prédit (tous points)"
        ))
else:
    # seulement la zone de recouvrement
    if not df_overlap.empty:
        fig.add_trace(go.Scatter(
            x=df_overlap["date_mesure"], y=df_overlap["predicted_taux_de_charge"],
            mode="lines", name="Prédit (overlap)"
        ))

fig.update_layout(
    title=f"Comparaison Réel vs Prédit — {name_selected}",
    xaxis_title="Temps",
    yaxis_title="Taux de charge",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig, use_container_width=True)

# Optional Altair demo
if show_altair and not df_real_sel.empty:
    try:
        import altair as alt
        if show_full_pred and not df_pred_sel.empty:
            base_real = df_real_sel[["date_mesure", "taux_de_charge"]].rename(columns={"date_mesure":"time","taux_de_charge":"value"})
            base_real["type"] = "réel"
            base_pred = df_pred_sel[["predicted_for", "predicted_taux_de_charge"]].rename(columns={"predicted_for":"time","predicted_taux_de_charge":"value"})
            base_pred["type"] = "prédit (tous)"
            base = pd.concat([base_real, base_pred], ignore_index=True)
        else:
            base = df_overlap[["date_mesure","taux_de_charge","predicted_taux_de_charge"]].rename(columns={"date_mesure":"time"})
            base = base.melt(id_vars=["time"], value_vars=["taux_de_charge","predicted_taux_de_charge"],
                             var_name="type", value_name="value")
        line_alt = alt.Chart(base).mark_line().encode(x="time:T", y="value:Q", color="type:N").properties(title=f"Altair – {name_selected}")
        st.altair_chart(line_alt, use_container_width=True)
    except Exception as e:
        st.info(f"Altair indisponible: {e}")

# ----------------------
# Error distribution (sur recouvrement)
# ----------------------
st.subheader("📊 Distribution des erreurs (sur la zone de recouvrement)")
if df_overlap.empty:
    st.write("—")
else:
    df_err = df_overlap.copy()
    df_err["error"] = df_err["taux_de_charge"] - df_err["predicted_taux_de_charge"]
    hist = px.histogram(df_err, x="error", nbins=40, marginal="box", title="Erreur = Réel – Prédit")
    st.plotly_chart(hist, use_container_width=True)

# ----------------------
# Equipements avec dernière prédiction > 0.5
# ----------------------
st.subheader("🚨 Équipements dont la prédiction la plus récente > 0.5")
if not df_pred_r.empty:
    latest_pred = (
        df_pred_r.sort_values("predicted_for")
        .groupby("name")
        .tail(5)
        .sort_values("predicted_taux_de_charge", ascending=False)
    )
    flagged = latest_pred[latest_pred["predicted_taux_de_charge"] > 0.5].copy()
    flagged = flagged.rename(columns={"name": "names"})
else:
    # modification sur le name
    flagged = pd.DataFrame(columns=["names", "predicted_for", "predicted_taux_de_charge"])
st.dataframe(flagged[["names", "predicted_for", "predicted_taux_de_charge"]], use_container_width=True, height=260)

# ----------------------
# Heatmap corrélations (réel)
# ----------------------
st.subheader("🧭 Heatmap de corrélation (équipements)")
try:
    pivot = df_all_r.pivot_table(index="date_mesure", columns="names", values="taux_de_charge", aggfunc="mean")
    corr = pivot.corr(method="pearson").fillna(0.0)
    heat = px.imshow(
        corr,
        x=corr.columns, y=corr.index,
        title="Corrélation Pearson entre équipements (réel)",
        aspect="auto", origin="lower", text_auto=False, zmin=-1, zmax=1
    )
    st.plotly_chart(heat, use_container_width=True)
except Exception as e:
    st.info(f"Heatmap non disponible: {e}")

# ----------------------
# Raw data expanders
# ----------------------
with st.expander("🔎 Données sources (réel)"):
    st.dataframe(df_all_r.sort_values(["names", "date_mesure"]).head(2000), use_container_width=True)

with st.expander("🔎 Données sources (prédictions)"):
    st.dataframe(df_pred_r.sort_values(["name", "predicted_for"]).head(2000), use_container_width=True)

st.caption("© Dashboard OLT – conçu pour votre projet de prédiction de taux de charge (Streamlit).")

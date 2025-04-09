import streamlit as st
import json
import pandas as pd
import altair as alt
import os

# Configuration de la page
st.set_page_config(page_title="📊 Logs DQN Agent", layout="wide")
st.title("🤖 Tableau de bord DQN Agent RL")

# Charger le fichier JSON
log_file = "logs.json"

if not os.path.exists(log_file):
    st.error("Fichier de log non trouvé.")
    st.stop()

with open(log_file, "r") as f:
    logs = json.load(f)

# Extraction des données
rewards = logs.get("rewards", [])
entropy = logs.get("entropy", [])
log = logs.get("log", [])
final_state = logs.get("state", {})

# 📊 Afficher les métriques finales
st.subheader("📌 État final du portefeuille")
col1, col2, col3 = st.columns(3)
col1.metric("💼 Portefeuille", f"{final_state.get('portfolio', 0):,.2f} €")
col2.metric("💰 Cash", f"{final_state.get('cash', 0):,.2f} €")
col3.metric("📈 Position", f"{final_state.get('position', 0):.4f} unité(s)")

# 📉 Récompense par épisode
st.subheader("💡 Récompenses par épisode")
st.line_chart(rewards)

# 📊 Entropie (diversité des actions)
st.subheader("🧠 Entropie des actions (proxy)")
st.line_chart(entropy)

# 🔍 Historique détaillé
st.subheader("📜 Détail des actions (log)")

if log:
    df_log = pd.DataFrame(log)
    df_log["step"] = df_log["step"].astype(int)
    df_log["portfolio"] = df_log["portfolio"].astype(float)

    # Sélecteur d'action
    selected_action = st.selectbox("Filtrer par action :", ["Toutes"] + sorted(df_log["action"].unique().tolist()))

    if selected_action != "Toutes":
        df_log = df_log[df_log["action"] == selected_action]

    # Affichage de la table
    st.dataframe(df_log[["step", "action", "reward", "portfolio", "cash", "position"]], use_container_width=True)

    # 📈 Graphique de la valeur du portefeuille avec surimpression des actions
    st.subheader("📈 Valeur du portefeuille avec points d’action")

    chart_base = alt.Chart(df_log).mark_line().encode(
        x=alt.X("step:Q", title="Étape"),
        y=alt.Y("portfolio:Q", title="Valeur du portefeuille (€)"),
        tooltip=["step", "portfolio", "action"]
    )

    points = alt.Chart(df_log).transform_filter(
        alt.datum.action != "hold"
    ).mark_point(size=70).encode(
        x="step:Q",
        y="portfolio:Q",
        color=alt.Color("action:N", scale=alt.Scale(domain=["buy", "sell"], range=["green", "red"])),
        shape=alt.Shape("action:N", scale=alt.Scale(domain=["buy", "sell"], range=["triangle", "triangle-down"])),
        tooltip=["step", "action", "portfolio"]
    )

    st.altair_chart((chart_base + points).interactive(), use_container_width=True)

else:
    st.warning("Aucun log détaillé d'action trouvé.")

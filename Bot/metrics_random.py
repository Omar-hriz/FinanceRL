import streamlit as st
import json
import os
import pandas as pd
import altair as alt

st.set_page_config(page_title="📊 Logs Agent RL", layout="centered")
st.title("📈 Visualisation des résultats de l’agent RL")

# --- Choix du fichier de log
log_file = st.sidebar.selectbox("Sélectionnez un fichier JSON :", [
    f for f in os.listdir("./") if f.endswith(".json")
], index=0)

log_path = os.path.join("./", log_file)

# --- Fonction de validation des types
def validate_log_data(logs):
    try:
        rewards = [float(r) for r in logs.get("rewards", [])]
        portfolio = [float(v) for v in logs.get("portfolio_values", [])]
        return rewards, portfolio
    except Exception as e:
        raise ValueError(f"Erreur dans les données du log : {e}")

# --- Chargement des données
if os.path.exists(log_path):
    try:
        with open(log_path, "r") as f:
            logs = json.load(f)

        rewards, portfolio = validate_log_data(logs)
        actions = logs.get("actions", [])
        state = logs.get("state", {})
        history = logs.get("history", [])

        # --- Métriques finales
        st.subheader("💰 Portefeuille final")
        col1, col2, col3 = st.columns(3)
        col1.metric("Valeur", f"{state.get('portfolio', 0):,.2f} €")
        col2.metric("Cash", f"{state.get('cash', 0):,.2f} €")
        col3.metric("Position", f"{state.get('position', 0):,.2f} unités")

        # --- Courbe des récompenses
        st.subheader("📉 Récompenses")
        st.line_chart(rewards)

        # --- Courbe de la valeur du portefeuille
        st.subheader("📈 Portefeuille")
        st.line_chart(portfolio)

        # --- Historique des actions
        if history:
            st.subheader("📍 Actions BUY/SELL")
            df = pd.DataFrame({
                "step": range(len(portfolio)),
                "valeur": portfolio,
                "action": ["HOLD"] + [a[1].upper() for a in history][1:]
            })

            chart = alt.Chart(df).mark_line().encode(
                x="step:Q",
                y="valeur:Q",
                tooltip=["step", "valeur", "action"]
            )

            points = alt.Chart(df).transform_filter(
                alt.datum.action != "HOLD"
            ).mark_point(size=70).encode(
                x="step:Q",
                y="valeur:Q",
                color=alt.Color("action:N", scale=alt.Scale(domain=["BUY", "SELL"], range=["green", "red"])),
                shape=alt.Shape("action:N", scale=alt.Scale(domain=["BUY", "SELL"], range=["triangle", "triangle-down"])),
                tooltip=["step", "action", "valeur"]
            )

            st.altair_chart(chart + points, use_container_width=True)

        st.success("✅ Fichier chargé avec succès.")

    except ValueError as ve:
        st.error(f"⚠️ Problème dans le contenu du fichier : {ve}")
    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")
else:
    st.error("❌ Fichier de log introuvable.")

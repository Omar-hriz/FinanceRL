import streamlit as st
import json
import os
import pandas as pd
import altair as alt

st.set_page_config(page_title="📊 Résultats RL Trading", layout="centered")
st.title("📈 Résultats de l’agent de trading")

log_path = "logs.json"

data_path = "data/AAPL.csv"

if os.path.exists(log_path):
    with open(log_path, "r") as f:
        data = json.load(f)

    rewards = data.get("rewards", [])
    entropy = data.get("entropy", [])
    last_state = data.get("state", {})

    st.subheader("💰 Portefeuille final")
    col1, col2, col3 = st.columns(3)
    col1.metric("Valeur", f"{last_state.get('portfolio', 0):.2f} €")
    col2.metric("Cash", f"{last_state.get('cash', 0):.2f} €")
    col3.metric("Position", f"{last_state.get('position', 0):.2f} unités")

    st.subheader("📉 Récompense totale par épisode")
    st.line_chart(rewards)

    st.subheader("📊 Entropie des actions (proxy)")
    st.line_chart(entropy)

    # 🔍 Affichage amélioré des données d'entraînement avec date
    if os.path.exists(data_path):
        st.subheader("📈 Donnée utilisée pour l'entraînement")
        df = pd.read_csv(data_path)

        # S'assurer qu'on a une colonne Date utilisable
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Close:Q", title="Prix de clôture"),
                tooltip=["Date", "Close", "Volume"]
            ).properties(
                width=700,
                height=300,
                title="Évolution du prix de clôture"
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("La colonne 'Date' est absente du fichier de données.")

    st.success("✅ Visualisation chargée avec succès.")
else:
    st.error("Aucun fichier 'logs.json' trouvé. Veuillez lancer un entraînement d'abord.")

import streamlit as st
import json
import os
import pandas as pd
import altair as alt

st.set_page_config(page_title="📊 Résultats RL Trading", layout="centered")
st.title("📈 Résultats de l’agent de trading")

import streamlit as st
import pandas as pd
import altair as alt
import json
import os

# Fichiers utilisés
log_path = "logs.json"
data_path = "data/comodity egg.xlsx"  # ⬅️ Remplace ici par sugar, rice, chili, etc.

if os.path.exists(log_path):
    with open(log_path, "r") as f:
        data = json.load(f)

    rewards = data.get("rewards", [])
    entropy = data.get("entropy", [])
    last_state = data.get("state", {})

    st.title("📊 Résultats de l'agent sur une commodité")

    st.subheader("💰 Portefeuille final")
    col1, col2, col3 = st.columns(3)
    col1.metric("Valeur", f"{last_state.get('portfolio', 0):.2f} €")
    col2.metric("Cash", f"{last_state.get('cash', 0):.2f} €")
    col3.metric("Position", f"{last_state.get('position', 0):.2f} unités")

    st.subheader("📉 Récompense totale par épisode")
    st.line_chart(rewards)

    st.subheader("📊 Entropie des actions (proxy)")
    st.line_chart(entropy)

    # 🔍 Affichage de la courbe de prix de la commodité utilisée
    if os.path.exists(data_path):
        st.subheader("📈 Prix de la commodité pendant l'entraînement")
        df = pd.read_excel(data_path)
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])
        df = df.rename(columns={"Kota Semarang": "Close","Tanggal":"Datetime"})
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X("Datetime:T", title="Date"),
                y=alt.Y("Close:Q", title="Prix"),
                tooltip=["Datetime", "Close"]
            ).properties(
                width=700,
                height=300,
                title="Évolution du prix de la commodité"
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("La colonne 'Datetime' est absente du fichier de données.")
else:
    st.error("Aucun fichier de log trouvé. Veuillez entraîner un agent d'abord.")

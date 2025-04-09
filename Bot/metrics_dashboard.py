import streamlit as st
import pandas as pd
import altair as alt
import json
import os

st.set_page_config(page_title="📊 Résultats RL Trading", layout="centered")
st.title("📈 Résultats de l’agent de trading")

# 📂 Fichiers utilisés
log_path = "logs_egg.json"
test_result_path = "data/results_test.json"
data_path = "data/comodity egg.xlsx"  # ⬅️ à adapter selon la commodité

# ------------------------------
# 🔁 Logs d'entraînement
# ------------------------------
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        data = json.load(f)

    rewards = data.get("rewards", [])
    losses = data.get("losses", [])
    last_state = data.get("state", {})

    st.subheader("📚 Entraînement de l'agent")
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Valeur finale", f"{last_state.get('portfolio', 0):.2f} €")
    col2.metric("💵 Cash", f"{last_state.get('cash', 0):.2f} €")
    col3.metric("📦 Position", f"{last_state.get('position', 0):.2f} unités")

    st.markdown("### 🎯 Récompense totale par épisode")
    st.line_chart(rewards)

    st.markdown("### 📉 Pertes (loss) pendant l'entraînement")
    st.line_chart(losses)
else:
    st.warning("⚠️ Aucune donnée d'entraînement trouvée (logs_egg.json).")

# ------------------------------
# 🧪 Résultats de test
# ------------------------------
if os.path.exists(test_result_path):
    st.subheader("🧪 Résultats sur les données de test")

    with open(test_result_path, "r") as f:
        test_data = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Portefeuille final", f"{test_data['final_portfolio']:.2f} €")
    col2.metric("📊 ROI", f"{test_data['roi']:.2f} %")
    col3.metric("⚖️ Sharpe Ratio", f"{test_data['sharpe_ratio']:.4f}")

    st.markdown("### 💸 Évolution du portefeuille (test)")
    st.line_chart(test_data["portfolio_evolution"])

else:
    st.warning("⚠️ Aucune donnée de test trouvée (results_test.json).")

# ------------------------------
# 📈 Données utilisées
# ------------------------------
if os.path.exists(data_path):
    st.subheader("🌾 Prix de la commodité utilisée")

    df = pd.read_excel(data_path)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.rename(columns={"Kota Semarang": "Close", "Tanggal": "Datetime"})

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Datetime:T", title="Date"),
            y=alt.Y("Close:Q", title="Prix (IDR)"),
            tooltip=["Datetime", "Close"]
        )
        .properties(
            width=700,
            height=300,
            title="📉 Évolution du prix de la commodité"
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.warning("⚠️ Fichier de données non trouvé.")

import streamlit as st
import pandas as pd
import altair as alt
import json
import os

st.set_page_config(page_title="📊 Comparaison Agents RL vs Hazard", layout="centered")
st.title("🤖 Agent RL vs 🎲 Agent Aléatoire")

# Onglets pour chaque agent
tab1, tab2 = st.tabs(["🤖 Agent RL (Robot)", "🎲 Agent Aléatoire (Hazard)"])

# ---------- AGENT RL ----------
with tab1:
    st.header("🤖 Résultats de l'agent RL")

    log_path = "logs_train_egg.json"
    test_result_path = "data/results_test.json"
    data_path = "data/comodity egg.xlsx"

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            data = json.load(f)

        rewards = data.get("rewards", [])
        losses = data.get("losses", [])
        last_state = data.get("state", {})

        st.subheader("📚 Entraînement")
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

    if os.path.exists(test_result_path):
        st.subheader("🧪 Résultats de test")

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
            .properties(width=700, height=300)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

# ---------- AGENT RANDOM ----------
with tab2:
    st.header("🎲 Résultats de l'agent Aléatoire")

    log_path = "logs_random.json"
    test_path = "data/results_test_random.json"

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)

        rewards = logs.get("rewards", [])
        portfolio = logs.get("portfolio_values", [])
        actions = logs.get("actions", [])
        state = logs.get("state", {})
        history = logs.get("history", [])

        st.subheader("📚 Logs aléatoires")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Valeur finale", f"{state.get('portfolio', 0):,.2f} €")
        col2.metric("💵 Cash", f"{state.get('cash', 0):,.2f} €")
        col3.metric("📦 Position", f"{state.get('position', 0):,.2f} unités")


        st.markdown("### 📈 Portefeuille")
        st.line_chart(portfolio)

        if history:
            st.markdown("### 🟢🛑 Actions BUY/SELL")
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
                shape=alt.Shape("action:N",
                                scale=alt.Scale(domain=["BUY", "SELL"], range=["triangle", "triangle-down"])),
                tooltip=["step", "action", "valeur"]
            )

            st.altair_chart(chart + points, use_container_width=True)
    else:
        st.warning("⚠️ Aucun log trouvé pour l'agent aléatoire.")

    if os.path.exists(test_path):
        with open(test_path, "r") as f:
            test_random = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("📈 Portefeuille final", f"{test_random['final_portfolio']:.2f} €")
        col2.metric("📊 ROI", f"{test_random['roi']:.2f} %")
        col3.metric("⚖️ Sharpe Ratio", f"{test_random['sharpe_ratio']:.4f}")

        st.markdown("### 💸 Évolution du portefeuille (random)")
        st.line_chart(test_random["portfolio_evolution"])
    else:
        st.warning("⚠️ Aucune donnée de test trouvée pour le random agent.")

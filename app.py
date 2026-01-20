import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.graph_objects as go
import yfinance as yf
from src.strategy_manager import StrategyManager

# --- KONFIGURÁCIÓ ---
CONFIG_FILE = 'strategies.json'
st.set_page_config(page_title="Crypto Bot Control", page_icon="💵", layout="wide")

# --- CSS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


# --- SEGÉDFÜGGVÉNYEK ---
def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {"swing": False, "daytrade": False, "pair": False}
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"swing": False, "daytrade": False, "pair": False}


def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)


# --- INIT MANAGER (Csak egyszer) ---
if 'manager' not in st.session_state:
    with st.spinner('Rendszer csatlakoztatása...'):
        try:
            st.session_state.manager = StrategyManager()
        except Exception as e:
            st.error(f"Hiba a manager indításakor: {e}")

if 'manager' in st.session_state:
    manager = st.session_state.manager
else:
    st.stop()

# --- OLDALSÁV (VEZÉRLŐPULT) ---
with st.sidebar:
    st.title("Távirányító")
    st.markdown("### Stratégiák Indítása")

    #Konfiguráció betöltése
    current_config = load_config()


    #Callback függvény: Ez fut le AZONNAL kattintáskor
    def update_config():
        new_config = {
            "swing": st.session_state['swing_key'],
            "daytrade": st.session_state['day_key'],
            "pair": st.session_state['pair_key']
        }
        save_config(new_config)



    st.toggle(
        "Swing Trade",
        value=current_config.get('swing', False),
        key="swing_key",
        on_change=update_config
    )

    st.toggle(
        "Daytrade",
        value=current_config.get('daytrade', False),
        key="day_key",
        on_change=update_config
    )

    st.toggle(
        "Pair Trading",
        value=current_config.get('pair', False),
        key="pair_key",
        on_change=update_config
    )

    st.markdown("---")

    # ÉLŐ EGYENLEG
    st.markdown("### Élő Egyenleg")
    try:
        bal = manager.exchange.fetch_balance()
        usdt = bal['USDT']['free']
        btc = bal['BTC']['free']
        eth = bal['ETH']['free']

        st.metric("USDT (Szabad)", f"${usdt:,.2f}")
        st.metric("BTC Pozíció", f"{btc:.5f} BTC")
        st.metric("ETH Pozíció", f"{eth:.5f} ETH")

    except Exception as e:
        st.error("Nem sikerült lekérni az egyenleget.")

    # Kézi frissítés gomb
    if st.button("Frissítés most"):
        st.rerun()

# --- FŐOLDAL TARTALMA ---
st.title("Monitorozó Központ")

# Élő Státusz Kijelző
live_config = load_config()

col1, col2, col3 = st.columns(3)
with col1:
    if live_config.get('swing'):
        st.success("Swing Trade: **FUT**")
    else:
        st.warning("Swing Trade: **ÁLL**")

with col2:
    if live_config.get('daytrade'):
        st.success("Daytrade: **FUT**")
    else:
        st.warning("Daytrade: **ÁLL**")

with col3:
    if live_config.get('pair'):
        st.success("Pair Trading: **FUT**")
    else:
        st.warning("Pair Trading: **ÁLL**")

st.markdown("---")

# NAPLÓ ÉS EREDMÉNYEK
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Élő Kereskedési Napló")
    if os.path.exists('trade_history.csv'):
        # Csak olvassuk, ha van adat
        try:
            df = pd.read_csv('trade_history.csv')
            if not df.empty:
                st.dataframe(df.sort_index(ascending=False), height=400, use_container_width=True)
            else:
                st.info("A napló üres.")
        except:
            st.warning("A napló fájl olvasása közben hiba történt.")
    else:
        st.info("Még nincs kötés a rendszerben.")

with col_right:
    st.subheader("Nyitott Pozíciók (JSON)")
    if os.path.exists('bot_state_multi.json'):
        try:
            with open('bot_state_multi.json', 'r') as f:
                state = json.load(f)
            st.json(state)
        except:
            st.write("Állapotfájl olvasása sikertelen.")
    else:
        st.write("A bot még nem inicializált.")

# --- AUTOMATIKUS FRISSÍTÉS (AUTO-REFRESH) ---
# Ez 5 másodpercenként frissíti az oldalt, hogy lásd a változásokat
time.sleep(5)
st.rerun()
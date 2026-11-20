"""Python entry point for the chosen RNN's HTML application."""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "app_site/dist/standalone.html"

st.set_page_config(page_title="Brent · Observatorio del petróleo", page_icon="🛢️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding:0;max-width:100%}header[data-testid='stHeader']{display:none}"
            "[data-testid='stAppViewContainer']{background:#0b1622;color:#e4edf5}iframe{border:0;width:100%}</style>", unsafe_allow_html=True)
if not PAGE.exists():
    st.error("Falta la interfaz construida. Ejecuta los siguientes comandos desde la raíz del proyecto.")
    st.code("python -m scripts.export_app\npython -m scripts.build_app", language="powershell")
    st.stop()

st.iframe(PAGE, width="stretch", height=1600, tab_index=0, alt="Observatorio del Brent: pronóstico, evaluación, contexto y datos")

import streamlit as st


xgboost_model= st.Page("xgboost_model.py", title=" 🌟 Stellar Classifier (XGBoost)")
autoencoder_model= st.Page("autoencoder_model.py", title=" 🧠 Light Curve Reconstructor Autoencoder")

pg = st.navigation([xgboost_model,autoencoder_model])

st.set_page_config(layout="wide")

pg.run()
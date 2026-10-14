"""Streamlit dashboard for API health and model monitoring."""
import json
import os

import pandas as pd
import requests
import streamlit as st

try:
    API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))
except FileNotFoundError:
    API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="MLOps Monitor", page_icon="📈", layout="wide")
st.title("Production ML Lifecycle Monitor")
st.caption("Breast cancer classification · reference versus sample production data")
st.info("Educational demo using a public dataset. This is not a medical tool or diagnosis.")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.success(f"API {health['status']} · model loaded: {health['model_loaded']}")
    st.metric("Predictions served", health["predictions_served"])
    if not health["model_loaded"]:
        st.info("Train the model with `python -m mlops_app.training`, then restart the API.")
    else:
        snapshot = requests.get(f"{API_URL}/monitoring", timeout=10).json()
        if not snapshot.get("ready"):
            st.info(snapshot.get("message", "Monitoring data is not ready."))
        else:
            drift, performance = snapshot["drift"], snapshot.get("performance", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Drift status", "Detected" if drift["drift_detected"] else "Clear")
            col2.metric("Mean PSI", f"{drift['mean_psi']:.3f}")
            col3.metric("Rows scored", snapshot["rows_scored"])
            if performance:
                st.metric("Sample production accuracy", f"{performance['accuracy']:.3f}",
                          delta=f"threshold {performance['min_accuracy']:.2f}")
            st.subheader("Prediction distribution")
            st.bar_chart(pd.Series(snapshot["prediction_counts"], name="count"))
            st.subheader("Feature drift (PSI)")
            psi = pd.Series(drift["feature_psi"], name="PSI").sort_values(ascending=False)
            st.bar_chart(psi.head(12))
            st.caption(f"Drift threshold: {drift['threshold']:.2f}. Drifted: {', '.join(drift['drifted_features']) or 'none'}")
            if snapshot["retraining_recommended"]:
                st.warning("Retraining recommended: drift or performance threshold was exceeded.")
            else:
                st.success("Current monitoring thresholds are satisfied.")
except requests.RequestException as exc:
    st.error(f"Cannot reach API at {API_URL}: {exc}")

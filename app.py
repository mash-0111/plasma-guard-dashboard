import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="PLASMA-GUARD", layout="wide")
st.title("🔬 PLASMA-GUARD: Density Limit Disruption Precursor Detection")

@st.cache_data
def load_data():
    df = pd.read_csv('dashboard_data.csv.gz')
    return df

@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

feature_cols = ['density', 'greenwald_fraction', 'elongation',
                 'minor_radius', 'plasma_current', 'toroidal_B_field',
                 'triangularity']

shot_ids = sorted(df['discharge_ID'].unique())
selected_shot = st.selectbox("اختر تجربة (Discharge ID):", shot_ids)

shot_data = df[df['discharge_ID'] == selected_shot].sort_values('time').reset_index(drop=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Greenwald Fraction عبر الزمن")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(shot_data['time'], shot_data['greenwald_fraction'], label='Greenwald fraction')
    ax.axhline(1.0, color='red', linestyle='--', label='Greenwald limit')
    precursor_times = shot_data[shot_data['density_limit_phase'] == 1]['time']
    if len(precursor_times) > 0:
        ax.axvspan(precursor_times.min(), precursor_times.max(), color='orange', alpha=0.3, label='Precursor phase')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Greenwald fraction')
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("تنبؤ الموديل عند آخر نقطة")
    last_point = shot_data.iloc[-1]
    X_input = last_point[feature_cols].values.reshape(1, -1)
    proba = model.predict_proba(X_input)[0, 1]

    st.metric("احتمال Precursor", f"{proba:.2%}")

    greenwald_val = last_point['greenwald_fraction']
    threshold = 0.47
    k = 10
    f_physics = 1 / (1 + np.exp(-k * (greenwald_val - threshold)))
    pcs = 1 - abs(proba - f_physics)

    st.metric("Physics Consistency Score", f"{pcs:.2f}")

    if pcs > 0.9:
        st.success("✅ اتفاق قوي بين الموديل والفيزياء")
    elif pcs > 0.7:
        st.warning("⚠️ اتفاق متوسط")
    else:
        st.error("🔴 اختلاف — يحتاج مراجعة بشرية")

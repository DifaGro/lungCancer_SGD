import streamlit as st
import joblib
from pathlib import Path
import numpy as np

# Tentukan path file secara dinamis
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "lung_cancer_sgd_model.pkl"
scaler_path = base_dir / "scaler.pkl"

# Load model dan scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Tampilan aplikasi
st.markdown("<h2>Prediksi Kanker Paru-Paru - Menggunakan Stochastic Gradient Descent</h2>", unsafe_allow_html=True)

# Membagi layar menjadi 2 kolom
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin:", ['Pria', 'Wanita'])
    smoking = st.selectbox("Apakah Anda Merokok?", ['Tidak', 'Ya'])
    yellow_fingers = st.selectbox("Apakah Anda Memiliki Jari Kuning?", ['Tidak', 'Ya'])
    anxiety = st.selectbox("Apakah Anda Merasakan Kecemasan?", ['Tidak', 'Ya'])
    chronic_disease = st.selectbox("Apakah Anda Memiliki Penyakit Kronis?", ['Tidak', 'Ya'])
    allergy = st.selectbox("Apakah Anda Memiliki Alergi?", ['Tidak', 'Ya'])
    coughing = st.selectbox("Apakah Anda Mengalami Batuk?", ['Tidak', 'Ya'])
    chest_pain = st.selectbox("Apakah Anda Mengalami Nyeri Dada?", ['Tidak', 'Ya'])

with col2:
    age = st.number_input("Usia:", min_value=1, max_value=100, step=1)
    peer_pressure = st.selectbox("Apakah Anda Merasakan Tekanan Sosial?", ['Tidak', 'Ya'])
    fatigue = st.selectbox("Apakah Anda Merasakan Kelelahan?", ['Tidak', 'Ya'])
    wheezing = st.selectbox("Apakah Anda Mengalami Gejala Mengi?", ['Tidak', 'Ya'])
    alcohol = st.selectbox("Apakah Anda Mengonsumsi Alkohol?", ['Tidak', 'Ya'])
    shortness_of_breath = st.selectbox("Apakah Anda Mengalami Sesak Napas?", ['Tidak', 'Ya'])
    swallowing_difficulty = st.selectbox("Apakah Anda Mengalami Kesulitan Menelan?", ['Tidak', 'Ya'])

# Konversi input ke format numerik sesuai model
gender = 2 if gender == 'Pria' else 1
smoking = 2 if smoking == 'Ya' else 1
yellow_fingers = 2 if yellow_fingers == 'Ya' else 1
anxiety = 2 if anxiety == 'Ya' else 1
peer_pressure = 2 if peer_pressure == 'Ya' else 1
chronic_disease = 2 if chronic_disease == 'Ya' else 1
fatigue = 2 if fatigue == 'Ya' else 1
allergy = 2 if allergy == 'Ya' else 1
wheezing = 2 if wheezing == 'Ya' else 1
alcohol = 2 if alcohol == 'Ya' else 1
coughing = 2 if coughing == 'Ya' else 1
shortness_of_breath = 2 if shortness_of_breath == 'Ya' else 1
swallowing_difficulty = 2 if swallowing_difficulty == 'Ya' else 1
chest_pain = 2 if chest_pain == 'Ya' else 1

# Buat array fitur (15 fitur)
features = np.array([[age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
                      allergy, wheezing, alcohol, coughing, shortness_of_breath, swallowing_difficulty,
                      chest_pain, gender]])

# Normalisasi data
features_scaled = scaler.transform(features)

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(features_scaled)
    result = "Kemungkinan Mengidap Kanker Paru-Paru" if prediction[0] == 2 else "Kemungkinan Tidak Mengidap Kanker Paru-Paru"
    st.write(f"Hasil Prediksi : {result}")

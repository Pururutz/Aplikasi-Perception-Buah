import streamlit as st
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier


def load_model():
    with open('buahPerception.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


def user_input_features():
    diameter = st.number_input('Diameter (cm)', min_value=0.0, value=10.0, step=0.1)
    weight = st.number_input('Weight (kg)', min_value=0.0, value=5.0, step=0.1)
    red = st.slider('Red Intensity (0-255)', min_value=0, max_value=255, value=120)
    green = st.slider('Green Intensity (0-255)', min_value=0, max_value=255, value=150)
    blue = st.slider('Blue Intensity (0-255)', min_value=0, max_value=255, value=200)

    data = {
        'Diameter': diameter,
        'Weight': weight,
        'Red': red,
        'Green': green,
        'Blue': blue
    }
    features = pd.DataFrame(data, index=[0])
    features.columns = features.columns.str.lower()
    return features


st.title('Aplikasi Prediksi Jenis Buah')

# Input dari pengguna
input_df = user_input_features()

# Menampilkan data input untuk memeriksa
st.write("Input Data Pengguna:")
st.write(input_df)

# Memuat model
model = load_model()

# Melakukan prediksi
prediction = model.predict(input_df)

# Menampilkan hasil prediksi
st.subheader('Hasil Prediksi Jenis Buah')

# Menampilkan hasil prediksi sebagai nama buah
predicted_species = prediction[0]  # Hasil prediksi berupa nama buah
st.write(f"Jenis buah yang diprediksi: {predicted_species}")
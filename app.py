import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
modelo = load_model('modelo.h5')

# Título de la aplicación
st.title("Clasificador de Imágenes con Deep Learning")

# Subir una imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    imagen = Image.open(uploaded_file)
    st.image(imagen, caption='Imagen subida', use_column_width=True)

    # Preprocesar la imagen
    imagen = imagen.resize((224, 224))
    imagen_array = np.array(imagen) / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)

    # Realizar la predicción
    prediccion = modelo.predict(imagen_array)
    etiqueta_predicha = np.argmax(prediccion, axis=1)

    # Mostrar el resultado
    st.write(f"Predicción: {etiqueta_predicha[0]}")

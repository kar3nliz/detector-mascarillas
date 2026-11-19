import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Detector de Mascarillas", page_icon="😷")
st.title("Detector de Mascarillas (Modelo TFLite)")
st.write("Sube una imagen para ejecutar la inferencia real del modelo.")

# Cargar modelo TFLite
@st.cache_resource
def cargar_modelo():
    interpreter = tf.lite.Interpreter(model_path="modelo_mascarillas.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = cargar_modelo()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Detectar dimensiones exactas automáticamente
    shape = input_details['shape']
    alto, ancho = int(shape), int(shape)
    st.success(f"✅ Modelo cargado correctamente (Entrada esperada: {ancho}x{alto} px)")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# Subir imagen
archivo = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="Imagen subida", width=300)
    
    # Preprocesar imagen al tamaño exacto del modelo
    img_redimensionada = imagen.resize((ancho, alto))
    datos_entrada = np.expand_dims(np.array(img_redimensionada, dtype=np.float32) / 255.0, axis=0)
    
    # Ejecutar inferencia
    interpreter.set_tensor(input_details['index'], datos_entrada)
    interpreter.invoke()
    salida = interpreter.get_tensor(output_details['index'])
    
    st.subheader("Resultados de la predicción:")
    for i, prob in enumerate(salida):
        st.write(f"**Clase {i}:** {prob * 100:.2f}%")

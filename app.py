import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

@st.cache_resource
def load_model():
    if not Path("model.h5").exists():
        return None
    return tf.keras.models.load_model("model.h5")

model = load_model()

st.title("🧬 Skin Cancer Detection")
st.write("Upload a skin lesion image to classify.")

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if model is None:
    st.warning("Model file `model.h5` is missing. Add it to the app directory to enable predictions.")
    st.stop()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    if prediction > threshold:
        st.error(f"Malignant ({prediction:.2f})")
    else:
        st.success(f"Benign ({1-prediction:.2f})")

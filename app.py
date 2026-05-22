import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from database import get_recent_predictions, init_db, save_prediction


init_db()

CLASS_LABELS = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]

@st.cache_resource
def load_model():
    model_path = Path("model.h5")
    if not model_path.exists():
        return None

    try:
        import tensorflow as tf
    except ImportError:
        return "tensorflow_missing"

    try:
        return tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as error:
        st.error(f"Unable to load model: {error}")
        return None

model = load_model()

st.title("🧬 Skin Cancer Detection")
st.write("Upload a skin lesion image to classify.")

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

with st.sidebar.expander("Recent Predictions"):
    recent_predictions = get_recent_predictions()
    if recent_predictions:
        for item in recent_predictions:
            st.write(f"{item['result']} - {item['filename']}")
            st.caption(f"Score: {item['prediction_score']:.2f} | {item['created_at']}")
    else:
        st.caption("No saved predictions yet.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if model is None:
    st.warning("Model file `model.h5` is missing or could not be loaded. Add a valid model file to the app directory to enable predictions.")
    st.stop()
elif model == "tensorflow_missing":
    st.warning("TensorFlow is not installed. Install dependencies with `pip install -r requirements.txt` to enable model loading.")
    st.stop()


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Unable to open the uploaded file as an image. Upload a valid JPG or PNG image.")
    else:
        st.image(image, caption="Uploaded Image")
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0]
        confidence = float(np.max(prediction))
        class_index = int(np.argmax(prediction))
        result = CLASS_LABELS[class_index] if class_index < len(CLASS_LABELS) else f"Class {class_index}"

        if confidence >= threshold:
            st.success(f"{result} ({confidence:.2f})")
        else:
            st.warning(f"{result} ({confidence:.2f}) is below the confidence threshold.")

        save_prediction(uploaded_file.name, confidence, threshold, result)

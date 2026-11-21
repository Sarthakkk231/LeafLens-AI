import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# =========================
# PAGE CONFIG (MUST BE FIRST st.* CALL)
# =========================
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")  # [web:23][web:24]

# =========================
# PATHS (USE EITHER ABSOLUTE OR RELATIVE, NOT BOTH)
# =========================
# Option A (recommended for your current setup): use absolute paths safely with raw strings. [web:28]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "class_names.json")

# Option B (when deploying): put files in ./model/ and use:
# MODEL_DIR = "model"
# MODEL_PATH = os.path.join(MODEL_DIR, "plant_resnet50_retrained.keras")
# LABELS_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL + LABELS
# =========================
@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(LABELS_PATH):
        st.error(f"❌ Labels file not found: {LABELS_PATH}")
        st.stop()

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        st.stop()

    # Ensure class_names is a list
    if isinstance(class_names, dict):
        # If it's {"0":"label0", ...} convert to list in correct order
        keys = list(class_names.keys())
        if all(str(k).isdigit() for k in keys):
            inv = {int(k): v for k, v in class_names.items()}
            n = max(inv.keys()) + 1
            class_names = [inv[i] for i in range(n)]
        else:
            # If it's {"label":0, ...}
            inv = {int(v): k for k, v in class_names.items()}
            n = max(inv.keys()) + 1
            class_names = [inv[i] for i in range(n)]

    class_names = [str(x) for x in class_names]
    return model, class_names

model, class_names = load_model_and_labels()

# =========================
# UI
# =========================
st.title("🌿 Plant Disease Classifier")
st.markdown("Upload an image of a plant leaf to classify potential diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image to get a prediction.")
    st.stop()

# Display uploaded image
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)

# Preprocess
img_array = np.array(image.resize(IMG_SIZE), dtype=np.float32)  # keep 0..255 for preprocess_input [web:32]
img_array = np.expand_dims(img_array, axis=0)                   # (1, 224, 224, 3)

# IMPORTANT: Use the preprocess that matches your training.
# For ResNet50 ImageNet-style preprocessing: [web:32]
processed_img = tf.keras.applications.resnet50.preprocess_input(img_array)

# Predict
with st.spinner("Classifying..."):
    preds = model.predict(processed_img, verbose=0)[0]

# Safety: label/model mismatch check
if len(preds) != len(class_names):
    st.error(
        f"❌ Mismatch: model outputs {len(preds)} classes, "
        f"but class_names has {len(class_names)} labels."
    )
    st.stop()

pred_idx = int(np.argmax(preds))
pred_name = class_names[pred_idx]
confidence = float(np.max(preds)) * 100

st.success(f"Prediction: **{pred_name}**")
st.write(f"Confidence: **{confidence:.2f}%**")

# Top-5
st.subheader("Top 5 Predictions")
top5 = np.argsort(preds)[::-1][:5]
for i in top5:
    st.write(f"- {class_names[int(i)]}: {float(preds[int(i)]) * 100:.2f}%")

st.markdown("---")
st.markdown("Developed using TensorFlow/Keras and Streamlit")

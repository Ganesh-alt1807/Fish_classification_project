import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model("models/mobilenetv2_best.h5")

class_names = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
               'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel',
               'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
               'fish sea_food sea_bass', 'fish sea_food shrimp',
               'fish sea_food striped_red_mullet', 'fish sea_food trout']

LABEL_MAP = {
    'animal fish': 'Animal Fish',
    'animal fish bass': 'Bass (Freshwater)',
    'fish sea_food black_sea_sprat': 'Black Sea Sprat',
    'fish sea_food gilt_head_bream': 'Gilt-Head Bream',
    'fish sea_food hourse_mackerel': 'Horse Mackerel',
    'fish sea_food red_mullet': 'Red Mullet',
    'fish sea_food red_sea_bream': 'Red Sea Bream',
    'fish sea_food sea_bass': 'Sea Bass',
    'fish sea_food shrimp': 'Shrimp',
    'fish sea_food striped_red_mullet': 'Striped Red Mullet',
    'fish sea_food trout': 'Trout'
}

st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image and get prediction")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=400)

    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    pred_label_raw = class_names[np.argmax(preds)]
    pred_label = LABEL_MAP.get(pred_label_raw, pred_label_raw)
    confidence = np.max(preds) * 100

    st.success(f"Prediction: {pred_label}")
    st.info(f"Confidence: {confidence:.2f}%")
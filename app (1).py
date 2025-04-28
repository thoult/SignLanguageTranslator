import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Background (Optional) ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1590402494682-0c3b8d4e1f4a");
background-size: cover;
background-repeat: no-repeat;
background-position: center;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Sidebar Info ---
st.sidebar.title("About This App")
st.sidebar.info(
    "This app translates Sign Language gestures into English text using a CNN model! 📚✨"
)

# --- Main Heading ---
st.title("🤟 Sign Language to Text Translator")

st.markdown(
    """
    Welcome to the **Sign Language Translator**!  
    Upload your gesture image below and click Translate! 🎯
    """
)

# --- Load Trained Model ---
@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.models.load_model('model_new.h5')
    return model

model = load_model()

# --- Upload Image Section ---
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

# --- Prediction Function ---
def predict_image(img):
    img = img.resize((64, 64))  # Adjust according to your model input
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, 64, 64, 3)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# --- After Upload ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Translate Sign'):
        with st.spinner('Translating... please wait 🌀'):
            result = predict_image(image)
            st.success(f"Predicted Class: **{result}**")

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ by YourName")


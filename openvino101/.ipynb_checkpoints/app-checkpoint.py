import streamlit as st
import PIL
import cv2
import numpy as np
import utils

st.set_page_config(
    page_title="Age/Gender/Emotion",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Age/Gender/Emotion :sun_with_face:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20))/100

input = None
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold = conf_threshold)
        
        st.image(visualized_image, channels = "BGR")
import streamlit as st
import PIL

st.set_page_config(
    page_title="Age/Gender/Emotion",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Age/Gender/Emotion :sun_with_face:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

input = None
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        st.image(uploaded_image)
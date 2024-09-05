import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set the page configuration
st.set_page_config(
    page_title="Face Anti-Spoofing Detection",
    page_icon=":face_with_monocle:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Function to load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('C:/Users/user/Desktop/FYP/3d_face_spoofing/notebooks/mobilenet/face_antispoofing_model_lcc_fasd_corrected_distr_40.keras')
    return model

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert to RGB in case of RGBA or other modes
    image = image.convert('RGB')
    img = image.resize((224, 224))  # Resize to the model's expected input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def make_prediction(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Header section
st.title("Face Anti-Spoofing Detection :detective:")

st.markdown("""
    Welcome to the **Face Anti-Spoofing Detection** app. 
    This tool uses a deep learning model to differentiate between real and spoofed faces.
    Upload an image, and the model will predict whether it's a real face or a spoof attempt.
""")

# File uploader
st.markdown("### Upload an Image")
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

# Display the uploaded image and make predictions
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add a separator
    st.markdown("---")
    
    # Prediction button
    st.markdown("### Predict the Result")
    if st.button("Predict"):
        with st.spinner("Analyzing the image..."):
            prediction = make_prediction(image, model)
        
        # Display the prediction result
        if prediction > 0.5:
            st.error("**Result: Spoof Detected**")
        else:
            st.success("**Result: Real Face Detected**")
    
    # Add some spacing
    st.markdown(" ")
else:
    st.info("Please upload an image file to begin.")

# Footer
st.markdown("---")
st.markdown("""
    *This application is developed as a part of a face anti-spoofing project. The model is based on the MobileNetV2 architecture.*  
    **Disclaimer:** This is a demo application. The model's performance might vary depending on the quality of the uploaded image.
""")

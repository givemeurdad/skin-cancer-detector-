import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
@st.cache_resource
def load_skin_model():
    model = load_model("skin_cancer_model.h5")
    return model

model = load_skin_model()

# Preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# UI
st.title("🧪 Skin Cancer Risk Estimator")
st.markdown("Upload a photo of a skin lesion or mole to get an **estimated risk**. This is not a diagnosis.")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Analyze
    with st.spinner("Analyzing..."):
        input_tensor = preprocess_image(image)
        prediction = model.predict(input_tensor)[0][0]  # Assumes binary classification

        # Convert prediction to risk level
        if prediction < 0.4:
            risk = "Low"
            color = "🟢"
        elif prediction < 0.7:
            risk = "Medium"
            color = "🟡"
        else:
            risk = "High"
            color = "🔴"

    st.subheader(f"{color} Risk Level: **{risk}**")
    st.write(f"Model confidence: `{prediction:.2f}`")

    st.warning("⚠️ This tool does not replace a professional diagnosis. Please consult a dermatologist for medical advice.")

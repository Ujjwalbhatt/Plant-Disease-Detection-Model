import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import streamlit as st
import time
import pandas as pd
import plotly.express as px

model = tf.keras.models.load_model('plant_disease_model.h5')

disease_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

recommendations = {
    "Apple___Apple_scab": "Apply appropriate fungicides and remove fallen infected leaves.",
    "Apple___Black_rot": "Prune and destroy infected branches and fruits. Use fungicides.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees or apply fungicides to prevent infection.",
    "Apple___healthy": "No action needed. Your apple plant is healthy.",
    "Blueberry___healthy": "No action needed. Your blueberry plant is healthy.",
    "Cherry_(including_sour)___Powdery_mildew": "Use fungicides and ensure good air circulation around plants.",
    "Cherry_(including_sour)___healthy": "No action needed. Your cherry plant is healthy.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply fungicides and practice crop rotation.",
    "Corn_(maize)___Common_rust_": "Use resistant varieties and apply fungicides if necessary.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant varieties and apply fungicides during the early stage of the disease.",
    "Corn_(maize)___healthy": "No action needed. Your corn plant is healthy.",
    "Grape___Black_rot": "Apply fungicides and remove and destroy infected fruits.",
    "Grape___Esca_(Black_Measles)": "Prune and destroy infected vines. Ensure proper vineyard sanitation.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Use fungicides and avoid overhead watering.",
    "Grape___healthy": "No action needed. Your grape plant is healthy.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove and destroy infected trees. Control insect vectors such as psyllids.",
    "Peach___Bacterial_spot": "Apply bactericides and ensure proper sanitation practices.",
    "Peach___healthy": "No action needed. Your peach plant is healthy.",
    "Pepper,_bell___Bacterial_spot": "Use copper-based bactericides and ensure good sanitation.",
    "Pepper,_bell___healthy": "No action needed. Your bell pepper plant is healthy.",
    "Potato___Early_blight": "Apply fungicides and practice crop rotation.",
    "Potato___Late_blight": "Apply fungicides and practice crop rotation.",
    "Potato___healthy": "No action needed. Your potato plant is healthy.",
    "Raspberry___healthy": "No action needed. Your raspberry plant is healthy.",
    "Soybean___healthy": "No action needed. Your soybean plant is healthy.",
    "Squash___Powdery_milde": "Use fungicides and ensure good air circulation around plants.",
    "Strawberry___Leaf_scorch": "Remove infected leaves and apply appropriate fungicides.",
    "Strawberry___healthy": "No action needed. Your strawberry plant is healthy.",
    "Tomato___Bacterial_spot": "Use bactericides and ensure proper sanitation and crop rotation.",
    "Tomato___Early_blight": "Apply fungicides and practice crop rotation.",
    "Tomato___Late_blight": "Apply fungicides and practice crop rotation.",
    "Tomato___Leaf_Mold": "Ensure good ventilation and apply fungicides if necessary.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves and apply fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use miticides and ensure proper irrigation and humidity control.",
    "Tomato___Target_Spot": "Apply fungicides and practice crop rotation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly populations and use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "Practice good sanitation and use virus-free seeds.",
    "Tomato___healthy": "No action needed. Your tomato plant is healthy."
}

def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_disease(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    disease_name = disease_classes[predicted_class]
    return disease_name, confidence, predictions[0]

# Streamlit Interface
def main():
    st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")
    
    st.markdown("""
    <style>
    p{
        font-size: 1.125rem !important;
    }
    .big-font {
        font-size: 3rem !important;
        font-weight:bold;
        color: #86d72f;
    }
    .stButton>button {
        font-size: 2rem;
        padding: 0.75rem 2rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        transition: background-color 0.3s, transform 0.3s;
    }
   .stButton>button:hover,
.stButton>button:focus {
         background-color: white;
        color: #4CAF50!important;
        border: none;
            transform: scale(1.05);
    }
   
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="big-font">Plant Disease Detection🌿</h1>', unsafe_allow_html=True)

    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses machine learning to detect plant diseases from images. "
        "Upload a clear image of a plant leaf to get started."
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            img = image.load_img(uploaded_file, target_size=(150, 150))
            st.image(img, caption='Uploaded Image', use_column_width=True)

        with col2:
            if st.button('Detect Disease'):
                img_array = preprocess_image(img)
                
                with st.spinner('Analyzing image...'):
                    disease_name, confidence, all_predictions = predict_disease(img_array)
                
                st.success(f"Detected Disease: {disease_name}")
                st.info(f"Confidence: {confidence:.2%}")
                
                recommendation = recommendations.get(disease_name, "No specific recommendation available.")
                st.warning(f"Recommendation: {recommendation}")

                # Display confidence scores
                df = pd.DataFrame({'Disease': disease_classes, 'Confidence': all_predictions})
                df = df.sort_values('Confidence', ascending=False).head(5)  # Top 5 predictions
                fig = px.bar(df, x='Disease', y='Confidence', title='Top 5 Prediction Confidences')
                st.plotly_chart(fig)
            

    st.markdown("## How to use")
    st.write("""
    1. Upload a clear image of a plant leaf
    2. Click on 'Detect Disease'
    3. View the detected disease, confidence, and recommendations
    4. Check the confidence scores for different predictions
    """)

    st.markdown("## Disclaimer")
    st.info(
        "This tool is for educational purposes only. "
        "Always consult with a professional botanist or plant pathologist for accurate diagnosis and treatment."
    )

if __name__ == "__main__":
    main()
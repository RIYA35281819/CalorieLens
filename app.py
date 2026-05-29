import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="CalorieLens", page_icon="🍎", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .title { text-align: center; color: #2e7d32; font-size: 3em; font-weight: bold; }
    .subtitle { text-align: center; color: #555; font-size: 1.2em; }
    .footer { text-align: center; color: #aaa; margin-top: 50px; font-size: 0.9em; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🍎 CalorieLens</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Food Label Nutrition Extractor</div>', unsafe_allow_html=True)
st.markdown("---")

# Upload
uploaded_file = st.file_uploader("📸 Upload a Food Label Image", type=["jpg", "png", "jpeg"])

def extract_nutrition(text):
    nutrition = {}
    patterns = {
        "Calories": r'calori[e]?s?\s*[:\-]?\s*(\d+)',
        "Fat": r'total\s*fat\s*[:\-]?\s*(\d+)',
        "Saturated_Fat": r'saturated\s*fat\s*[:\-]?\s*(\d+)',
        "Cholesterol": r'cholesterol\s*[:\-]?\s*(\d+)',
        "Sodium": r'sodium\s*[:\-]?\s*(\d+)',
        "Carbohydrate": r'total\s*carb\w*\s*[:\-]?\s*(\d+)',
        "Fiber": r'dietary\s*fiber\s*[:\-]?\s*(\d+)',
        "Protein": r'protein\s*[:\-]?\s*(\d+)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text.lower())
        if match:
            nutrition[key] = int(match.group(1))
    return nutrition

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Label", use_container_width=True)

    with col2:
        st.info("⏳ Scanning label...")
        reader = easyocr.Reader(['en'])
        result = reader.readtext(np.array(image))
        extracted_text = " ".join([text[1] for text in result])
        nutrition = extract_nutrition(extracted_text)

        if nutrition:
            st.success("✅ Nutrition Info Found!")
            df = pd.DataFrame(nutrition.items(), columns=["Nutrient", "Amount"])
            st.table(df)
        else:
            st.warning("⚠️ Could not extract info. Try a clearer image!")

    if nutrition:
        st.markdown("---")

        # ML Prediction
        st.subheader("🤖 ML Health Prediction")
        input_data = [
            nutrition.get("Calories", 0),
            nutrition.get("Fat", 0),
            nutrition.get("Saturated_Fat", 0),
            nutrition.get("Cholesterol", 0),
            nutrition.get("Sodium", 0),
            nutrition.get("Carbohydrate", 0),
            nutrition.get("Fiber", 0),
            nutrition.get("Protein", 0),
        ]
        prediction = model.predict([input_data])[0]
        confidence = model.predict_proba([input_data]).max() * 100

        if prediction == "Healthy":
            st.success(f"✅ This food is **HEALTHY**! (Confidence: {confidence:.1f}%)")
        else:
            st.error(f"❌ This food is **UNHEALTHY**! (Confidence: {confidence:.1f}%)")

        # Chart
        st.markdown("---")
        st.subheader("📊 Nutrition Breakdown")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#2e7d32','#66bb6a','#a5d6a7','#ff7043','#ffa726','#42a5f5','#ab47bc','#26c6da']
        ax.bar(nutrition.keys(), nutrition.values(), color=colors[:len(nutrition)])
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.ylabel("Amount")
        plt.tight_layout()
        st.pyplot(fig)

        # Calorie meter
        st.markdown("---")
        calories = nutrition.get("Calories", 0)
        st.subheader(f"🔥 Calories: {calories} kcal")
        if calories < 100:
            st.success("Low calorie food! Great choice 💚")
        elif calories < 300:
            st.warning("Moderate calories 🟡")
        else:
            st.error("High calorie food! Consume in moderation 🔴")

st.markdown('<div class="footer">Made with ❤️ by CalorieLens</div>', unsafe_allow_html=True)
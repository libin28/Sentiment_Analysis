import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit app UI
st.title("🧠 Sentiment Analysis App")
st.write("Enter some text below to analyze its sentiment.")

# User input
user_input = st.text_area("Your Text", placeholder="Type your text here...")

# Prediction
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)[0]

        # Display sentiment
        if prediction == 1:
            st.success("✅ Positive Sentiment")
        elif prediction == 0:
            st.info("😐 Neutral Sentiment")
        else:
            st.error("❌ Negative Sentiment")

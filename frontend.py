import streamlit as st
import requests

st.title("Sentiment Analysis Demo")

# Text input
user_input = st.text_area("Enter text to analyze:")

# Button to submit
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Send POST request to FastAPI
        url = "http://127.0.0.1:8000/predict"
        response = requests.post(url, json={"texts": [user_input]})
        result = response.json()

        # Display prediction
        prediction = result["predictions"][0]
        st.subheader("Predictions:")
        st.write(f"Logistic Regression: {prediction['logistic_regression']}")
        st.write(f"LightGBM: {prediction['lightgbm']}")

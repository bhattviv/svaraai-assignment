# Sentiment Analysis of Email Replies

This project provides a machine learning and NLP pipeline to classify email replies into **positive**, **negative**, or **neutral** sentiments. It includes a **FastAPI-based API** for real-time predictions.

---

## **Project Structure**

Deploy-BERT-for-Sentiment-Analysis-with-FastAPI/
│
├─ sentiment_analyzer/
│ └─ classifier/
│ ├─ baseline.py # Baseline models (Logistic Regression, LightGBM)
│ ├─ label_mapping.pkl # Label mapping for predictions
│ ├─ tfidf_vectorizer.pkl
│ └─ baseline_lgbm_model.pkl / baseline_logreg_model.pkl
│
├─ app.py # FastAPI application
├─ frontend.py # Streamlit interactive demo
├─ test_model.py # Script to test models locally
├─ reply_classification_dataset.csv
├─ README.md # This file
└─ requirements.txt # Python dependencies




## **Setup Instructions**

### **1. Clone the repository**

git clone <your-repo-url>
cd Deploy-BERT-for-Sentiment-Analysis-with-FastAPI

2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux / Mac

3. Install dependencies
pip install -r requirements.txt

Running the API (FastAPI)

Start the FastAPI server:

python -m uvicorn app:app --reload


Open Swagger UI to test endpoints:

http://127.0.0.1:8000/docs


GET /: Check if API is running.

POST /predict: Send JSON input to get sentiment predictions.

Example JSON input:

{
  "texts": [
    "Looking forward to the demo!",
    "I am not interested in this offer."
  ]
}


Example JSON output:

{
  "predictions": [
    {
      "text": "Looking forward to the demo!",
      "logistic_regression": "positive",
      "lightgbm": "positive"
    },
    {
      "text": "I am not interested in this offer.",
      "logistic_regression": "negative",
      "lightgbm": "negative"
    }
  ]
}

Running the Streamlit Demo

Open a new terminal (FastAPI server must be running).

Run:

python -m streamlit run frontend.py


A browser window will open with a text box. Enter your text and click Predict Sentiment.

Testing Models Locally

You can use test_model.py to quickly check predictions without running the API:

python test_model.py

Requirements

Python 3.10+

Libraries: pandas, numpy, scikit-learn, lightgbm, joblib, fastapi, uvicorn, streamlit

Install all dependencies using pip install -r requirements.txt.

Project Highlights

Baseline models: Logistic Regression and LightGBM with TF-IDF vectorization.

FastAPI /predict endpoint supports batch predictions.

Streamlit demo for interactive sentiment analysis.

Saved models and vectorizer for reproducibility.

Next Steps / Improvements

Fine-tune a transformer model (e.g., DistilBERT) for better performance.

Include confidence scores in API responses.

Deploy the app online using Render, Railway, or Hugging Face Spaces.

Add a Dockerfile for containerized deployment.

Author

Vivek Bhatt
AI/ML Intern | Data Science Enthusiast


---

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Dict
import torch  # If you're using PyTorch
import lightgbm as lgb  # Assuming you're using LightGBM or other model libraries
from sentiment_analyzer.classifier.model import Model
from sentiment_analyzer.classifier.model import Model, get_model

app = FastAPI()

# Define the request body structure
class SentimentRequest(BaseModel):
    text: str

# Define the response body structure
class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float

# Predict endpoint
@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: Model = Depends(get_model)):
    """
    Takes a text input, runs the sentiment analysis model, and returns the sentiment and probabilities.
    """
    # Assuming model.predict() gives the raw probabilities or logits from the model
    raw_probs = model.predict(request.text)  # You should replace this with the actual prediction logic
    
    # Get the sentiment with the highest probability (assuming model output is raw probabilities)
    sentiment = "positive" if torch.argmax(raw_probs) == 1 else "negative"  # Example logic for binary sentiment
    confidence = float(torch.max(raw_probs))  # Convert PyTorch tensor to a float for confidence

    # Format probabilities to be a dictionary of string keys and float values
    probabilities = {str(i): float(prob) for i, prob in enumerate(raw_probs.tolist())}

    # Return the response as an instance of SentimentResponse
    return SentimentResponse(
        probabilities=probabilities,
        sentiment=sentiment,
        confidence=confidence
    )


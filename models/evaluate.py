from transformers import pipeline
import os

# -----------------------------
# Load fine-tuned model
# -----------------------------
def load_model():

    # Get absolute project path (prevents Streamlit path issues)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "model", "saved_model")

    # Load Hugging Face pipeline
    classifier = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path
    )

    return classifier


# -----------------------------
# Single prediction (Chatbot use)
# -----------------------------
def predict_sentiment(model, text):

    result = model(text)[0]

    label = result["label"]
    score = result["score"]

    # Normalize Hugging Face outputs
    if label.upper() in ["POSITIVE", "LABEL_1", "1"]:
        sentiment = "positive"
    else:
        sentiment = "negative"

    return sentiment, float(score)


# -----------------------------
# Batch prediction (Dashboard use)
# -----------------------------
def predict_batch(model, texts):

    predictions = []

    for text in texts:

        result = model(text)[0]

        label = result["label"]
        score = result["score"]

        if label.upper() in ["POSITIVE", "LABEL_1", "1"]:
            sentiment = "positive"
        else:
            sentiment = "negative"

        predictions.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": float(score)
        })

    return predictions
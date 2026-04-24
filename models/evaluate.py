from transformers import pipeline

def load_model():

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return classifier



def predict_sentiment(model, text):

    result = model(text)[0]

    label = result["label"]
    score = result["score"]


    if label.upper() == "POSITIVE":
        sentiment = "positive"
    else:
        sentiment = "negative"

    return sentiment, float(score)



def predict_batch(model, texts):

    predictions = []

    for text in texts:

        result = model(text)[0]

        label = result["label"]
        score = result["score"]

        if label.upper() == "POSITIVE":
            sentiment = "positive"
        else:
            sentiment = "negative"

        predictions.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": float(score)
        })

    return predictions
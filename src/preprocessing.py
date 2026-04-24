import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_data(df):

    df["text"] = df["text"].str.lower()

    
    df["text"] = df["text"].str.translate(str.maketrans('', '', string.punctuation))

    
    df["tokens"] = df["text"].str.split()

    
    stop_words = set(stopwords.words("english"))
    df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if word not in stop_words])

    
    lemmatizer = WordNetLemmatizer()
    df["tokens"] = df["tokens"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return df
import pandas as pd
import matplotlib.pyplot as plt

from .preprocessing import preprocess_data

def load_data():
    df = pd.read_csv("data/data.csv", on_bad_lines='skip')
    df = preprocess_data(df)

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["month"] = df["date"].dt.strftime("%B")
    return df

def filter_data(df, source=None, sentiment=None, month=None):
    filtered_df = df.copy()

    if source != "All":
        filtered_df = filtered_df[filtered_df["source"] == source]

    if sentiment != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == sentiment]

    if month is not None and month != "All":
        filtered_df = filtered_df[filtered_df["month"] == month]

    return filtered_df


def create_dashboard(df):
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    counts = df["sentiment"].value_counts()
    plt.bar(counts.index, counts.values, color="skyblue",width=0.6)
    plt.title("Sentiment Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Count", fontsize=12)


    plt.subplot(2, 2, 2) 
    counts = df["sentiment"].value_counts()
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    plt.title("Sentiment Share",fontsize=14, fontweight="bold")
    
    
    plt.subplot(2, 2, 3)
    avg_rating = df.groupby("source")["rating"].mean()
    plt.bar(avg_rating.index, avg_rating.values, width=0.6)
    plt.title("Rating by Source")
    plt.xlabel("Source")
    plt.ylabel("Avg Rating")
    plt.xticks(rotation=30)
    
    
    plt.subplot(2, 2, 4)
    counts = df["location"].value_counts()
    plt.bar(counts.index, counts.values, color="green")
    plt.title("Reviews by Location", fontsize=14, fontweight="bold")
    plt.xlabel("Location", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()

    return plt



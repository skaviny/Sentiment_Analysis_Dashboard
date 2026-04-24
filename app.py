import streamlit as st

from src.eda import (
    load_data,
    filter_data,
    create_dashboard
)

from models.evaluate import load_model, predict_sentiment


st.set_page_config(page_title="Sentiment Dashboard", layout="wide")


df = load_data()
model = load_model()


st.sidebar.title("Control Panel")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Chatbot"]
)


if page == "Dashboard":

    st.title("SENTIMENT ANALYSIS DASHBOARD")

    st.sidebar.title("Filters")

    source = st.sidebar.selectbox(
        "Select Source",
        ["All"] + list(df["source"].unique())
    )

    sentiment_filter = st.sidebar.selectbox(
        "Select Sentiment",
        ["All"] + list(df["sentiment"].unique())
    )
    
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    months = sorted(df["month"].unique(), key=lambda x: month_order.index(x) if x in month_order else 999)
    
    month_filter = st.sidebar.selectbox(
        "Select Month",
        ["All"] + months
    )

    
    filtered_df = filter_data(df, source, sentiment_filter, month_filter)

    
    col1, col2, col3, col4 = st.columns(4)

    st.markdown("---")

    total = len(filtered_df)
    positive = len(filtered_df[filtered_df["sentiment"] == "positive"])
    negative = len(filtered_df[filtered_df["sentiment"] == "negative"])

    col1.metric("Total Reviews", total)
    col2.metric("Positive", positive)
    col3.metric("Negative", negative)


    st.pyplot(create_dashboard(filtered_df), clear_figure=True)

    st.markdown("---")

    st.subheader("Key Insights")

    
    total = len(filtered_df)
    positive = len(filtered_df[filtered_df["sentiment"] == "positive"])
    negative = len(filtered_df[filtered_df["sentiment"] == "negative"])

    pos_per = (positive / total) * 100 if total > 0 else 0
    neg_per = (negative / total) * 100 if total > 0 else 0

    top_location = filtered_df["location"].value_counts().idxmax()


    avg_rating = filtered_df["rating"].mean()

    top_rated_source = filtered_df.groupby("source")["rating"].mean().idxmax()

    st.success(f"📌 Positive: {pos_per:.1f}%")
    st.error(f"⚠️ Negative: {neg_per:.1f}%")
    st.info(f"🌍 Top Location: {top_location}")
    st.info(f"⭐ Avg Rating: {avg_rating:.2f}")
    st.info(f"🏆 Highest Rated Source: {top_rated_source}")


elif page == "Chatbot":

    st.sidebar.empty()

    st.title("🤖 Live Sentiment Prediction")

    user_input = st.text_area("Enter your text here")

    if st.button("Predict Sentiment"):
        if user_input.strip() != "":
            pred_sentiment, confidence = predict_sentiment(model, user_input)

            if pred_sentiment == "positive":
                st.success(f"✅ Sentiment: {pred_sentiment.upper()}")
            else:
                st.error(f"❌ Sentiment: {pred_sentiment.upper()}")
            
            st.info(f"Confidence Score: {confidence:.2f}")
        else:
            st.warning("Please enter text to analyze")


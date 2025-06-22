import os
import time
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import streamlit as st
import pandas as pd
import altair as alt
from textblob import TextBlob
import re

# Gemini API Setup (if available)
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    genai_enabled = True
except Exception as e:
    print("❌ Gemini not available:", e)
    genai_enabled = False

# --- Fetch from DuckDuckGo ---
def fetch_duckduckgo_news(company, limit=10):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://duckduckgo.com/html/?q={company}+news"
    articles = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("a", class_="result__a")
        for result in results:
            text = result.get_text(strip=True)
            if text and len(text.split()) > 6:
                articles.append((text, "DuckDuckGo News"))
                if len(articles) >= limit:
                    break
    except Exception as e:
        print("DuckDuckGo fetch error:", e)
    return articles

# --- Fetch from YouTube (via basic scraping fallback) ---
def fetch_youtube_titles(brand, max_results=5):
    titles = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://www.youtube.com/results?search_query={brand}"
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for video in soup.find_all("a", href=True):
            title = video.get("title")
            if title and brand.lower() in title.lower():
                titles.append((title.strip(), "YouTube"))
            if len(titles) >= max_results:
                break
        print(f"✅ {len(titles)} YouTube video titles scraped.")
    except Exception as e:
        print("YouTube fetch error:", e)
    return titles

# --- Sentiment Analysis Logic ---
def analyze_sentiment(text, use_gemini=False):
    text = re.sub(r"[^\w\s]", "", text).strip()  # normalize text
    if use_gemini and genai_enabled:
        try:
            prompt = f"Classify this comment as Positive, Negative or Neutral:\n'{text}'"
            response = model.generate_content(prompt)
            label = response.text.strip().split("\n")[0]
            if any(s in label.lower() for s in ["positive", "neutral", "negative"]):
                return label.title()
        except Exception as e:
            print("Gemini error:", e)
    # Fallback to TextBlob
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# --- Visualization & Dashboard ---
def analyze_and_visualize(brand):
    st.subheader("📊 Corporate Website Sentiment Analysis")
    st.write(f"Analyzing public perception of **{brand}** across web platforms...")

    with st.spinner("Fetching DuckDuckGo News..."):
        duck_texts = fetch_duckduckgo_news(brand)

    with st.spinner("Fetching YouTube video titles..."):
        youtube_texts = fetch_youtube_titles(brand)

    all_texts = duck_texts + youtube_texts
    if not all_texts:
        st.warning("❌ No data found for sentiment analysis.")
        return

    data = pd.DataFrame(all_texts, columns=["Text", "Source"])
    use_gemini = st.checkbox("Use Gemini for sentiment analysis (first 10 entries)", value=False)

    sentiment_list = []
    progress = st.progress(0)
    for i, row in data.iterrows():
        sentiment = analyze_sentiment(row["Text"], use_gemini and i < 10)
        sentiment_list.append(sentiment)
        progress.progress((i + 1) / len(data))
    data["Sentiment"] = sentiment_list

    # Sentiment Distribution
    sentiment_counts = data["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    # 📊 Bar Chart
    chart = alt.Chart(sentiment_counts).mark_bar().encode(
        x="Sentiment",
        y="Count",
        color="Sentiment",
        tooltip=["Sentiment", "Count"]
    ).properties(title="Sentiment Distribution")
    st.altair_chart(chart, use_container_width=True)

    # 🥧 Pie Chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts["Count"], labels=sentiment_counts["Sentiment"],
            autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1, use_container_width=True)

    # ☁️ Word Cloud
    combined_text = " ".join(data["Text"].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    st.image(wordcloud.to_array(), use_container_width=True, caption="Word Cloud of Comments")

    # 📋 Sentiment Table
    st.subheader("📋 Sentiment by Source")
    st.dataframe(data[["Source", "Sentiment", "Text"]])

    # 📊 Summary Table
    st.subheader("📊 Summary by Platform")
    summary_table = data.groupby(["Source", "Sentiment"]).size().unstack(fill_value=0)
    st.dataframe(summary_table)

    # 🧾 Legend
    st.subheader("🧾 What Each Sentiment Means")
    st.markdown("""
    | Sentiment | Description |
    |-----------|-------------|
    | ✅ Positive | Feedback with praise, support, trust |
    | ⚪ Neutral | Mixed or unclear sentiment |
    | ❌ Negative | Complaints, criticism, or concerns |
    """)

# --- Streamlit UI ---
st.set_page_config(page_title="Corporate Sentiment Analyzer", layout="centered")
st.title("🌐 Corporate Brand Sentiment Analyzer")

brand_input = st.text_input("Enter a brand or company name", "Broadridge")
if st.button("Analyze"):
    analyze_and_visualize(brand_input)

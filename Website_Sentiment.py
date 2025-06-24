import os
import re
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from textblob import TextBlob
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from datetime import datetime

# --- Gemini Setup ---
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    genai_enabled = True
except Exception:
    genai_enabled = False

# --- Fetch Functions ---
def fetch_duckduckgo_news(brand, limit=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://duckduckgo.com/html/?q={brand}+news"
    articles = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("a", class_="result__a")
        for result in results[:limit]:
            text = result.get_text(strip=True)
            if text:
                articles.append((text, "DuckDuckGo News", datetime.now().date()))
    except Exception as e:
        print("DuckDuckGo error:", e)
    return articles

def fetch_youtube_titles(brand, limit=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://duckduckgo.com/html/?q=site:youtube.com+{brand}"
    titles = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("a", class_="result__a")
        for res in results[:limit]:
            text = res.get_text(strip=True)
            if text:
                titles.append((text, "YouTube", datetime.now().date()))
    except Exception as e:
        print("YouTube error:", e)
    return titles

def fetch_twitter_titles(brand, limit=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://duckduckgo.com/html/?q=site:twitter.com+{brand}"
    tweets = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("a", class_="result__a")
        for res in results[:limit]:
            text = res.get_text(strip=True)
            if text:
                tweets.append((text, "Twitter", datetime.now().date()))
    except Exception as e:
        print("Twitter error:", e)
    return tweets

def fetch_glassdoor_reviews(brand, limit=5):
    reviews = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://duckduckgo.com/html/?q=site:glassdoor.com+{brand}+reviews"
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("a", class_="result__a")
        for res in results[:limit]:
            text = res.get_text(strip=True)
            if text:
                reviews.append((text, "Glassdoor", datetime.now().date()))
    except Exception as e:
        print("Glassdoor error:", e)
    return reviews

def fetch_linkedin_titles(brand, limit=5):
    return [(f"Simulated LinkedIn post about {brand} #{i+1}", "LinkedIn", datetime.now().date()) for i in range(limit)]

# --- Sentiment Analysis ---
def analyze_sentiment(text, use_gemini=False):
    clean_text = re.sub(r"[^\w\s]", "", text)
    if use_gemini and genai_enabled:
        try:
            prompt = f"Analyze and classify this text as Positive, Negative, or Neutral:\n'{clean_text}'"
            response = model.generate_content(prompt)
            label = response.text.strip().split("\n")[0]
            if any(x in label.lower() for x in ["positive", "neutral", "negative"]):
                return label.title()
        except Exception as e:
            print("Gemini error:", e)
    polarity = TextBlob(clean_text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    return "Neutral"

# --- Streamlit UI ---
st.set_page_config(page_title="Corporate Brand Analyzer", layout="centered")
st.title("🌐 Corporate Brand Sentiment Analyzer")

brand_input = st.text_input("Enter a company or brand name", "Broadridge")

if st.button("Analyze"):
    brand = brand_input

    with st.spinner("Fetching all data..."):
        duck = fetch_duckduckgo_news(brand)
        youtube = fetch_youtube_titles(brand)
        twitter = fetch_twitter_titles(brand)
        linkedin = fetch_linkedin_titles(brand)
        glassdoor = fetch_glassdoor_reviews(brand)

    all_data = duck + youtube + twitter + linkedin + glassdoor
    if not all_data:
        st.warning("❌ No data found.")
        st.stop()

    df = pd.DataFrame(all_data, columns=["Text", "Source", "Date"])
    use_gemini = st.checkbox("Use Gemini AI (for first 10 rows)", value=False)

    sentiment_list = []
    progress = st.progress(0)
    for i, row in df.iterrows():
        sent = analyze_sentiment(row["Text"], use_gemini and i < 10)
        sentiment_list.append(sent)
        progress.progress((i + 1) / len(df))
    df["Sentiment"] = sentiment_list

    st.subheader("📈 Sentiment Trend Over Time")
    trend = df.groupby(["Date", "Sentiment"]).size().reset_index(name="Count")
    st.altair_chart(
        alt.Chart(trend).mark_line(point=True).encode(
            x="Date:T", y="Count:Q", color="Sentiment:N"
        ), use_container_width=True
    )

    st.subheader("📊 Sentiment Distribution")
    st.altair_chart(
        alt.Chart(df["Sentiment"].value_counts().reset_index()).mark_bar().encode(
            x="index:N", y="Sentiment:Q", color="index:N"
        ).properties(title="Overall Sentiment")
    )

    st.subheader("📡 Source Contribution")
    source_counts = df['Source'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig, use_container_width=True)

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["Text"]))
    st.image(wordcloud.to_array(), use_container_width=True)

    if st.checkbox("Provide Feedback on Sentiment Accuracy"):
        df["Feedback"] = df.apply(lambda row:
            st.selectbox(f"{row['Text'][:70]}... Label: {row['Sentiment']}", ["Correct", "Incorrect", "Uncertain"],
                         key=row.name), axis=1)

    if genai_enabled and st.checkbox("🧠 Generate Executive Summary (AI)", value=False):
        with st.spinner("Generating summary..."):
            try:
                summary_input = "\n".join([f"{row['Source']}: {row['Text']}" for _, row in df.iterrows()][:30])
                prompt = f"""Give a concise executive summary of public sentiment for **{brand}** across platforms.
Data:
{summary_input}

End with 2 actionable recommendations."""
                response = model.generate_content(prompt)
                st.markdown(f"📝 **Executive Summary:**\n\n{response.text}")
            except Exception as e:
                st.error("❌ Gemini error")
                st.exception(e)

    st.subheader("📋 Sentiment Table")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV Report", csv, "sentiment_report.csv", "text/csv")

    st.subheader("🧾 What Each Sentiment Means")
    st.markdown("""
| Sentiment | Description |
|-----------|-------------|
| ✅ Positive | Praise, support, or trust |
| ⚪ Neutral   | Mixed, unclear, or factual |
| ❌ Negative | Complaints, criticism, or risk |
""")

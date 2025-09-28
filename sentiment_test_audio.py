import os
import asyncio
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
from pytube import YouTube
from pydub import AudioSegment
from docling import Docling

# Step 1: Download and convert YouTube to MP3
async def download_and_convert_audio(youtube_url, output_mp3='audio.mp3'):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    downloaded_file = stream.download(filename='temp_audio')
    print(f"✅ Downloaded: {downloaded_file}")

    audio = AudioSegment.from_file(downloaded_file)
    audio.export(output_mp3, format="mp3")
    print(f"🎵 Converted to MP3: {output_mp3}")

    os.remove(downloaded_file)
    return output_mp3

# Step 2: Perform sentiment analysis
async def SiteValidation():
    load_dotenv()
    API_KEY = os.getenv("ASSEMBLY_API_KEY")
    if not API_KEY:
        raise EnvironmentError("❌ ASSEMBLY_API_KEY not found in .env file.")

    docling = Docling(api_key=API_KEY)

    youtube_url = "https://www.youtube.com/watch?v=PHe0bXAIuk0"
    audio_path = await download_and_convert_audio(youtube_url)

    print("🧠 Transcribing with sentiment analysis...")
    doc = await docling.transcribe(audio_path, sentiment_analysis=True)

    if not doc.sentiment_analysis:
        print("⚠️ No sentiment data found in transcript.")
        return

    scores = [s["confidence"] for s in doc.sentiment_analysis]
    labels = [s["sentiment"] for s in doc.sentiment_analysis]

    if not scores:
        print("⚠️ Empty sentiment result.")
        return

    colors = ['green' if s == 'positive' else 'red' if s == 'negative' else 'yellow' for s in labels]

    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Bar chart
    plt.figure()
    plt.bar(range(len(scores)), scores, color=colors)
    plt.xticks(range(len(scores)), labels, rotation=45)
    plt.xlabel('Sentences')
    plt.ylabel('Confidence Score')
    plt.title("Sentiment Analysis (Docling)")

    bar_path = 'reports/sentiment_bar.png'
    plt.savefig(bar_path)
    print(f"📊 Bar chart saved to {bar_path}")
    plt.close()

    # Pie chart
    counter = Counter(labels)
    plt.figure()
    plt.pie(counter.values(), labels=counter.keys(), autopct='%1.1f%%')
    pie_path = 'reports/sentiment_pie.png'
    plt.savefig(pie_path)
    print(f"🥧 Pie chart saved to {pie_path}")
    plt.close()

if __name__ == "__main__":
    asyncio.run(SiteValidation())

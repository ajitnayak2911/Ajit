import os
import asyncio
import assemblyai as aai
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
from moviepy.editor import AudioFileClip
import yt_dlp

async def SiteValidation():
    """Download YouTube audio and run sentiment analysis."""
    # Load API key
    load_dotenv()
    API_KEY = os.getenv("ASSEMBLY_API_KEY")
    if not API_KEY:
        raise EnvironmentError("ASSEMBLY_API_KEY not found in .env file.")
    aai.settings.api_key = API_KEY

    # YouTube audio download
    youtube_url = "https://www.youtube.com/watch?v=TLu7C3RAGI8"
    audio_output = "downloaded_audio.m4a"
    print("📥 Downloading audio from YouTube...")

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio',
        'outtmpl': audio_output,
        'quiet': True,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    print(f"Audio downloaded: {audio_output}")

    # Convert to WAV using moviepy (AssemblyAI requires WAV/MP3)
    print("Converting to WAV...")
    wav_path = "converted_audio.wav"
    audio_clip = AudioFileClip(audio_output)
    audio_clip.write_audiofile(wav_path, logger=None)
    audio_clip.close()
    print(f"Audio converted to WAV: {wav_path}")

    # Transcribe and analyze sentiment
    print("Uploading to AssemblyAI...")
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(sentiment_analysis=True)
    transcript = transcriber.transcribe(wav_path, config)

    print("Analyzing sentiment...")
    scores = []
    labels = []
    for result in transcript.sentiment_analysis:
        scores.append(result.confidence)
        labels.append(result.sentiment)

    colors = ['green' if s == 'positive' else 'red' if s == 'negative' else 'yellow' for s in labels]

    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(scores)), scores, color=colors)
    plt.xticks(range(len(scores)), labels, rotation=45)
    plt.xlabel('Sentences')
    plt.ylabel('Confidence Score')
    plt.title('Sentiment Analysis of YouTube Audio')
    plt.tight_layout()
    bar_file = 'reports/sentiment_bar.png'
    plt.savefig(bar_file)
    print(f"Bar chart saved to {bar_file}")
    plt.close()

    # Pie chart
    counter = Counter(labels)
    plt.figure()
    plt.pie(counter.values(), labels=counter.keys(), autopct='%1.1f%%')
    pie_file = 'reports/sentiment_pie.png'
    plt.savefig(pie_file)
    print(f"Pie chart saved to {pie_file}")
    plt.close()

if __name__ == "__main__":
    asyncio.run(SiteValidation())

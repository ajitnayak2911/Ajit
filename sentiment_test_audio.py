import os
import assemblyai as aai
import matplotlib.pyplot as plt
from collections import Counter
import asyncio
from dotenv import load_dotenv

async def SiteValidation():
    """Perform sentiment analysis on an audio file and generate charts as files."""
    # Loading environment variables from .env
    load_dotenv()
    API_KEY = os.getenv("ASSEMBLY_API_KEY")
    if not API_KEY:
        raise EnvironmentError("ASSEMBLY_API_KEY not found in .env file.")
    aai.settings.api_key = API_KEY

    audio_file = "https://assembly.ai/wildfires.mp3"

    config = aai.TranscriptionConfig(sentiment_analysis=True)

    transcript = aai.Transcriber().transcribe(audio_file, config)

    scores = []
    labels = []
    for result in transcript.sentiment_analysis:
        scores.append(result.confidence)
        labels.append(result.sentiment)

    colors = ['green' if s == 'positive' else 'red' if s == 'negative' else 'yellow' for s in labels]

    if not os.path.exists('reports'):
        os.makedirs('reports')

    # Plotting the sentiment scores
    plt.figure()
    plt.bar(range(len(scores)), scores, color=colors)
    plt.xticks(range(len(scores)), labels, rotation=45)
    plt.xlabel('Sentences')
    plt.ylabel('Confidence Score')
    plt.title('Sentiment Analysis of Audio')
    bar_file = 'reports/sentiment_bar.png'
    plt.savefig(bar_file)
    print(f"✅ Bar chart successfully saved to {bar_file}")
    plt.close()

    # Plotting pie chart of sentiment proportions
    counter = Counter(labels)

    plt.figure()
    plt.pie(counter.values(), labels=counter.keys(), autopct='%1.1f%%')
    pie_file = 'reports/sentiment_pie.png'
    plt.savefig(pie_file)
    print(f"✅ Pie chart successfully saved to {pie_file}")
    plt.close()


if __name__ == "__main__":
    asyncio.run(SiteValidation())

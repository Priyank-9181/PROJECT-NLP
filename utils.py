import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import pandas as pd


def plot_emotion_distribution(df):
    """
    Create a bar plot showing the distribution of emotions in the dataset
    """
    fig = px.bar(
        df["emotion"].value_counts().reset_index(),
        x="index",
        y="emotion",
        title="Distribution of Emotions",
        labels={"index": "Emotion", "emotion": "Count"},
    )
    return fig


def create_wordcloud(texts, title):
    """
    Create a word cloud visualization for the given texts
    """
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(texts)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    return fig


def plot_frequent_words(texts, emotion, n=10):
    """
    Create a horizontal bar plot of most frequent words
    """
    words = " ".join(texts).split()
    word_freq = pd.Series(words).value_counts().head(n)
    fig = px.bar(
        x=word_freq.values,
        y=word_freq.index,
        orientation="h",
        title=f"Most Frequent Words in {emotion} Tweets",
        labels={"x": "Frequency", "y": "Words"},
    )
    return fig

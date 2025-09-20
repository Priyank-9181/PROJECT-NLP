import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import emoji
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import nltk

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# Load Hindi stopwords
def load_hindi_stopwords():
    try:
        with open("stop_hinglish.txt", "r", encoding="utf-8") as f:
            hindi_stops = set(f.read().splitlines())
        return hindi_stops
    except:
        return set()


class TweetPreprocessor:
    def __init__(self):
        self.hindi_stops = load_hindi_stopwords()
        self.english_stops = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Extract emojis and map them to text
        emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
        emoji_text = " ".join(
            [emoji.demojize(e).replace(":", " ").replace("_", " ") for e in emoji_list]
        )

        # Remove emojis from original text
        text = emoji.replace_emoji(text, "")

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Add emoji text back (if any)
        if emoji_text:
            text = f"{text} {emoji_text}"

        return text

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        words = [
            w
            for w in words
            if w not in self.english_stops and w not in self.hindi_stops
        ]
        return " ".join(words)

    def lemmatize(self, text):
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text


class EmotionAnalyzer:
    def __init__(self):
        self.preprocessor = TweetPreprocessor()
        self.vectorizer = None
        self.model = None

    def prepare_data(self, df):
        df["cleaned_text"] = df["Tweet_with_Emojis"].apply(self.preprocessor.preprocess)
        return df

    def extract_features(self, texts, train=False):
        if train:
            self.vectorizer = TfidfVectorizer(max_features=1000)
            features = self.vectorizer.fit_transform(texts)
        else:
            features = self.vectorizer.transform(texts)
        return features

    def train_model(self, X, y):
        self.model = SVC(kernel="linear")
        self.model.fit(X, y)
        return self.model

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)

    def predict(self, text):
        cleaned_text = self.preprocessor.preprocess(text)
        features = self.vectorizer.transform([cleaned_text])
        return self.model.predict(features)[0]

    def save_model(self, filepath="models/emotion_analyzer.pkl"):
        import os

        os.makedirs("models", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump((self.vectorizer, self.model), f)
        return filepath

    def load_model(self, filepath="models/emotion_analyzer.pkl"):
        with open(filepath, "rb") as f:
            self.vectorizer, self.model = pickle.load(f)
        return self


def train_and_save_model(data_path):
    """
    Train and save the emotion analyzer model
    """
    # Load data
    df = pd.read_csv(data_path)

    # Initialize analyzer
    analyzer = EmotionAnalyzer()

    # Preprocess data
    processed_df = analyzer.prepare_data(df)

    # Extract features
    X = analyzer.extract_features(processed_df["cleaned_text"], train=True)
    y = df["Emotion"]  # Use original column name from data.csv

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    analyzer.train_model(X_train, y_train)

    # Evaluate model
    evaluation_report = analyzer.evaluate_model(X_test, y_test)

    # Save model
    model_path = analyzer.save_model()

    return model_path, evaluation_report


if __name__ == "__main__":
    # Example usage
    data_path = "data.csv"  # Update with your data path
    model_path, eval_report = train_and_save_model(data_path)
    print(f"Model saved to: {model_path}")
    print("\nModel Evaluation Report:")
    print(eval_report)

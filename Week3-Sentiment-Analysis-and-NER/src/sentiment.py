from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sentiment_dataset.csv"

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

def load_data():
    data = pd.read_csv(DATA_PATH)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["text"] != ""]

    if data.empty:
        raise ValueError("The sentiment dataset does not contain usable rows.")

    return data["text"], data["label"]

def train_model():
    texts, labels = load_data()
    model = build_model()
    model.fit(texts, labels)
    return model

def predict_sentiment(text):
    if not text or not text.strip():
        raise ValueError("Please enter some text.")
    return train_model().predict([text])[0]

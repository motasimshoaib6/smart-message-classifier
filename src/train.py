
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "messages.csv"


def load_data():
    """Load messages and labels from the CSV file."""
    data = pd.read_csv(DATA_PATH)

    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("CSV must contain text and label columns.")

    if data["text"].isna().any() or data["label"].isna().any():
        raise ValueError("Dataset contains missing values.")

    return data


def build_model():
    """Create the text classification pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])


def main():
    data = load_data()

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n--- Model Evaluation ---")
    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("\n--- Test Your Own Messages ---")
    print("Type 'exit' to stop.")

    while True:
        message = input("\nEnter a message: ").strip()

        if message.lower() == "exit":
            break

        if not message:
            print("Please enter a message.")
            continue

        prediction = model.predict([message])[0]
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()

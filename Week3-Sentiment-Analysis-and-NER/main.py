import argparse

from src.sentiment import predict_sentiment
from src.ner import extract_entities


def main():
    parser = argparse.ArgumentParser(
        description="Week 3 NLP project: Sentiment Analysis and Named Entity Recognition"
    )
    parser.add_argument("--sentiment", help="Text to classify")
    parser.add_argument("--ner", help="Text from which to extract named entities")
    args = parser.parse_args()

    if args.sentiment is not None:
        print("Sentiment:", predict_sentiment(args.sentiment))
    elif args.ner is not None:
        print("Named entities:")
        entities = extract_entities(args.ner)
        if not entities:
            print("No entities found. Check that the spaCy English model is installed.")
        for entity in entities:
            print(
                f"  {entity['text']} — {entity['label']} "
                f"({entity['description']})"
            )
    else:
        print("Choose a task:")
        print("1. Sentiment Analysis")
        print("2. Named Entity Recognition (NER)")
        choice = input("Enter 1 or 2: ").strip()
        text = input("Enter text: ").strip()

        if choice == "1":
            print("Sentiment:", predict_sentiment(text))
        elif choice == "2":
            entities = extract_entities(text)
            if not entities:
                print("No entities found. Check that the spaCy English model is installed.")
            for entity in entities:
                print(
                    f"{entity['text']} — {entity['label']} "
                    f"({entity['description']})"
                )
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()

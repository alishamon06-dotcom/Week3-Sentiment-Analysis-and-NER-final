# Week 3 NLP Mini Project: Sentiment Analysis and Named Entity Recognition

**Submission name: Alisha Mondal**

This project demonstrates two introductory Natural Language Processing tasks:

1. **Sentiment analysis** – classifies a short text as positive, negative, or neutral.
2. **Named Entity Recognition (NER)** – identifies named entities in English text using spaCy.

## Project structure

```text
Week3-Sentiment-Analysis-and-NER/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── sentiment_dataset.csv
└── src/
    ├── __init__.py
    ├── sentiment.py
    ├── ner.py
    └── evaluate.py
```

## Technologies

- Python
- pandas
- scikit-learn
- spaCy (`en_core_web_sm`)

## Setup on Windows

Open PowerShell in the folder containing `main.py`, then run:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the project

Start the interactive menu:

```powershell
python main.py
```

Run sentiment analysis directly:

```powershell
python main.py --sentiment "I love this product. It works perfectly."
```

Run NER directly:

```powershell
python main.py --ner "Alisha visited Kolkata on Monday and met Google employees."
```

## Evaluate the sentiment model

Run this command from the project root:

```powershell
python -m src.evaluate
```

The included dataset contains 135 short examples, with 45 examples in each sentiment class. Using stratified five-fold cross-validation with a fixed random seed, the current run reports:

- Mean accuracy: **0.985**
- Mean macro F1-score: **0.985**

These figures describe this small demonstration dataset only. Many rows are close variations of a limited set of base sentences, so the scores should not be treated as evidence of strong performance on new, real-world text. The training-set classification report printed by the script is also not an independent test result.

## Notes and limitations

- The sentiment model uses character-level TF-IDF features and Logistic Regression.
- The classifier is trained from the included CSV file when a prediction is requested.
- NER uses spaCy's pretrained English model. It may miss entities, assign an incorrect label, or behave differently with unusual names and context.
- The dataset is small and simple. It does not adequately represent sarcasm, slang, regional language, code-mixed text, or a wide range of writing styles.
- The project is intended for learning and demonstration, not for production use.

No API key is required. The project runs locally after the dependencies and spaCy model are installed.

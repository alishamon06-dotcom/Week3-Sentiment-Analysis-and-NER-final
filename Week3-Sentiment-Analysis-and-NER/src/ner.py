def extract_entities(text):
    """Return entities found by spaCy's pretrained English model."""
    if not text or not text.strip():
        return []
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        print("spaCy or its English model is missing. Install with:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
        return []
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_, "description": spacy.explain(ent.label_) or ent.label_}
        for ent in doc.ents
    ]

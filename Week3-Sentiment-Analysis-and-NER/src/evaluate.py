from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

from src.sentiment import build_model, load_data


def main():
    texts, labels = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = build_model()

    results = cross_validate(
        model,
        texts,
        labels,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
    )

    print(
        "5-fold CV accuracy: "
        f"{results['test_accuracy'].mean():.3f} "
        f"(+/- {results['test_accuracy'].std():.3f})"
    )
    print(
        "5-fold CV macro F1: "
        f"{results['test_f1_macro'].mean():.3f} "
        f"(+/- {results['test_f1_macro'].std():.3f})"
    )

    model.fit(texts, labels)
    print("\nTraining-set report (not an independent test result):")
    print(classification_report(labels, model.predict(texts), zero_division=0))


if __name__ == "__main__":
    main()

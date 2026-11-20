
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_processed_news():
    """
    Load the two preprocessed news documents.
    """

    news_a_path = PROCESSED_DATA_DIR / "news_a_processed.txt"
    news_b_path = PROCESSED_DATA_DIR / "news_b_processed.txt"

    if not news_a_path.exists():
        raise FileNotFoundError(f"File not found: {news_a_path}")

    if not news_b_path.exists():
        raise FileNotFoundError(f"File not found: {news_b_path}")

    news_a = news_a_path.read_text(encoding="utf-8")
    news_b = news_b_path.read_text(encoding="utf-8")

    return [news_a, news_b]


def create_tfidf_vectors(documents):
    """
    Convert documents into TF-IDF vectors.
    """

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    return vectorizer, tfidf_matrix


def display_tfidf_results(vectorizer, tfidf_matrix):
    """
    Display basic information about the TF-IDF matrix
    and the most important terms in each document.
    """

    feature_names = vectorizer.get_feature_names_out()

    print("TF-IDF Results")
    print("=" * 60)

    print(f"Number of documents: {tfidf_matrix.shape[0]}")
    print(f"Number of unique terms: {tfidf_matrix.shape[1]}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    for document_index in range(tfidf_matrix.shape[0]):

        scores = tfidf_matrix[document_index].toarray().flatten()

        top_indices = scores.argsort()[::-1][:10]

        print(f"\nTop terms in News {document_index + 1}:")
        
        for index in top_indices:
            if scores[index] > 0:
                print(
                    f"{feature_names[index]}: "
                    f"{scores[index]:.4f}"
                )


def main():
    documents = load_processed_news()

    vectorizer, tfidf_matrix = create_tfidf_vectors(documents)

    display_tfidf_results(vectorizer, tfidf_matrix)


if __name__ == "__main__":
    main()

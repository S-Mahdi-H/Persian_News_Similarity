
import asyncio

from src.extraction.news_extractor import main as extract_news
from src.preprocessing.text_preprocessor import process_news

from src.feature_extraction.tfidf_vectorizer import (
    load_processed_news,
    create_tfidf_vectors,
    display_tfidf_results
)

from src.similarity.similarity_calculator import (
    calculate_cosine_similarity,
    calculate_euclidean_distance,
    display_results
)


def main():

    # Step 1: Extract news articles
    print("\n" + "=" * 60)
    print("STEP 1: NEWS EXTRACTION")
    print("=" * 60)

    asyncio.run(extract_news())

    # Step 2: Preprocess news articles
    print("\n" + "=" * 60)
    print("STEP 2: TEXT PREPROCESSING")
    print("=" * 60)

    process_news()

    # Step 3: Load processed documents
    print("\n" + "=" * 60)
    print("STEP 3: LOADING PROCESSED DOCUMENTS")
    print("=" * 60)

    documents = load_processed_news()

    if len(documents) != 2:
        raise ValueError("Exactly two news documents are required.")

    # Step 4: Extract TF-IDF features
    print("\n" + "=" * 60)
    print("STEP 4: TF-IDF FEATURE EXTRACTION")
    print("=" * 60)

    vectorizer, tfidf_matrix = create_tfidf_vectors(documents)

    display_tfidf_results(vectorizer, tfidf_matrix)

    # Step 5: Calculate similarity metrics
    print("\n" + "=" * 60)
    print("STEP 5: SIMILARITY MEASUREMENT")
    print("=" * 60)

    cosine_score = calculate_cosine_similarity(tfidf_matrix)

    euclidean_distance = calculate_euclidean_distance(tfidf_matrix)

    # Step 6: Display final results
    print("\n" + "=" * 60)
    print("STEP 6: FINAL RESULTS")
    print("=" * 60)

    display_results(cosine_score, euclidean_distance)


if __name__ == "__main__":
    main()

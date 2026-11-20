
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from src.feature_extraction.tfidf_vectorizer import (
    load_processed_news,
    create_tfidf_vectors
)


def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculate cosine similarity between two documents.
    """

    similarity = cosine_similarity(
        tfidf_matrix[0],
        tfidf_matrix[1]
    )[0][0]

    return similarity


def calculate_euclidean_distance(tfidf_matrix):
    """
    Calculate Euclidean distance between two documents.
    """

    distance = euclidean_distances(
        tfidf_matrix[0],
        tfidf_matrix[1]
    )[0][0]

    return distance


def display_results(cosine_score, euclidean_distance):
    """
    Display similarity measurement results.
    """

    print("=" * 60)
    print("NEWS SIMILARITY RESULTS")
    print("=" * 60)

    print(f"Cosine Similarity: {cosine_score:.6f}")
    print(f"Cosine Similarity (%): {cosine_score * 100:.2f}%")

    print(f"Euclidean Distance: {euclidean_distance:.6f}")

    print("=" * 60)


def main():

    # Load preprocessed news documents
    documents = load_processed_news()

    if len(documents) != 2:
        raise ValueError("Exactly two news documents are required.")

    # Generate TF-IDF vectors
    vectorizer, tfidf_matrix = create_tfidf_vectors(documents)

    # Calculate similarity metrics
    cosine_score = calculate_cosine_similarity(tfidf_matrix)

    euclidean_distance = calculate_euclidean_distance(tfidf_matrix)

    # Display results
    display_results(cosine_score, euclidean_distance)


if __name__ == "__main__":
    main()


from pathlib import Path

from src.preprocessing.basic_preprocessing import basic_preprocess
from src.preprocessing.morphological_preprocessing import morphological_preprocess
from src.preprocessing.stopword_removal import remove_stopwords


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def preprocess_text(text):
    """
    Apply the complete preprocessing pipeline.
    """

    # Step 1: Basic preprocessing
    basic_text = basic_preprocess(text)

    # Step 2: Morphological preprocessing
    morphological_tokens = morphological_preprocess(
        basic_text
    )

    # Step 3: Stopword removal
    final_tokens = remove_stopwords(
        morphological_tokens
    )

    return " ".join(final_tokens)


def process_news():
    """
    Process both news articles and save their results.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    news_files = [
        ("news_a.txt", "news_a_processed.txt"),
        ("news_b.txt", "news_b_processed.txt")
    ]

    for input_name, output_name in news_files:

        input_path = RAW_DATA_DIR / input_name
        output_path = PROCESSED_DATA_DIR / output_name

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        # Read raw news
        raw_text = input_path.read_text(
            encoding="utf-8"
        )

        # Preprocess news
        processed_text = preprocess_text(raw_text)

        # Save processed news
        output_path.write_text(
            processed_text,
            encoding="utf-8"
        )

        print(f"Processed: {input_name}")
        print(f"Original characters: {len(raw_text)}")
        print(f"Processed characters: {len(processed_text)}")
        print(f"Processed words: {len(processed_text.split())}")
        print(f"Saved to: {output_path}")
        print("-" * 60)


if __name__ == "__main__":
    process_news()

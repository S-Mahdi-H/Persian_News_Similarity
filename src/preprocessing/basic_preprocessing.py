
import re
import unicodedata


# Convert Arabic characters to Persian equivalents
CHARACTER_MAP = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
})


def normalize_text(text):
    """
    Normalize Persian characters and Unicode representation.
    """

    text = unicodedata.normalize("NFC", text)

    text = text.translate(CHARACTER_MAP)

    # Remove Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Remove Tatweel
    text = text.replace("\u0640", "")

    return text


def remove_punctuation(text):
    """
    Remove punctuation while preserving Persian letters,
    numbers, whitespace and ZWNJ.
    """

    text = re.sub(r"[^\w\s\u200c]", " ", text)

    # Remove underscores
    text = text.replace("_", " ")

    return text


def normalize_whitespace(text):
    """
    Replace multiple spaces with a single space.
    """

    return re.sub(r"\s+", " ", text).strip()


def basic_preprocess(text):
    """
    Apply all basic preprocessing steps.
    """

    text = normalize_text(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)

    return text

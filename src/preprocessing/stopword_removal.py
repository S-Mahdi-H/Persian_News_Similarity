from pathlib import Path

from src.preprocessing.basic_preprocessing import normalize_text


# Locate stopwords file
CURRENT_DIR = Path(__file__).resolve().parent
STOPWORDS_FILE = CURRENT_DIR / "stopwords_fa.txt"


def load_stopwords():
    """
    Load Persian stopwords from an independent text file.
    """

    if not STOPWORDS_FILE.exists():
        raise FileNotFoundError(
            f"Stopwords file not found: {STOPWORDS_FILE}"
        )

    words = STOPWORDS_FILE.read_text(
        encoding="utf-8"
    ).splitlines()

    return {
        normalize_text(word.strip())
        for word in words
        if word.strip()
    }


STOPWORDS = load_stopwords()


# Copular forms that should be removed
COPULA_LEMMAS = {
    "است",
    "هست",
}


# Future auxiliary forms
FUTURE_AUXILIARIES = {
    "خواهد",
    "خواهند",
    "خواهم",
    "خواهی",
    "خواهیم",
    "خواهید",
    "نخواهد",
    "نخواهند",
    "نخواهم",
    "نخواهی",
    "نخواهیم",
    "نخواهید",
}


def is_negative(surface, lemma):
    """
    Detect negative verb forms.

    Stanza may remove the negative prefix from the lemma.
    Therefore, we inspect the original surface form.
    """

    surface = normalize_text(surface)
    lemma = normalize_text(lemma)

    if surface.startswith("نمی"):
        return True

    if surface.startswith("ن") and not lemma.startswith("ن"):
        return True

    return False


def remove_stopwords(morphological_tokens):
    """
    Remove stopwords while preserving verbs and negation.
    """

    filtered_tokens = []

    index = 0

    while index < len(morphological_tokens):

        token = morphological_tokens[index]

        surface = normalize_text(token["surface"])
        lemma = normalize_text(token["lemma"])
        upos = token["upos"]

        # -------------------------------------------------
        # Step 1: Handle future auxiliary verbs
        # -------------------------------------------------

        if (
            upos == "AUX"
            and surface in FUTURE_AUXILIARIES
            and index + 1 < len(morphological_tokens)
        ):

            next_token = morphological_tokens[index + 1]

            if next_token["upos"] == "VERB":

                next_lemma = normalize_text(
                    next_token["lemma"]
                )

                negative = is_negative(surface, lemma)

                if negative:
                    filtered_tokens.append(
                        f"NEG_{next_lemma}"
                    )
                else:
                    filtered_tokens.append(next_lemma)

                # Skip both auxiliary and main verb
                index += 2
                continue

        # -------------------------------------------------
        # Step 2: Preserve verbs and their polarity
        # -------------------------------------------------

        if upos == "VERB":

            if is_negative(surface, lemma):
                filtered_tokens.append(
                    f"NEG_{lemma}"
                )
            elif lemma:
                filtered_tokens.append(lemma)

            index += 1
            continue

        # -------------------------------------------------
        # Step 3: Handle auxiliary verbs
        # -------------------------------------------------

        if upos == "AUX":

            if is_negative(surface, lemma):
                filtered_tokens.append(
                    f"NEG_{lemma}"
                )

            elif lemma not in COPULA_LEMMAS and lemma:
                filtered_tokens.append(lemma)

            index += 1
            continue

        # -------------------------------------------------
        # Step 4: Remove ordinary stopwords
        # -------------------------------------------------

        if surface in STOPWORDS or lemma in STOPWORDS:
            index += 1
            continue

        # Keep the normalized lemma
        if lemma:
            filtered_tokens.append(lemma)

        index += 1

    return filtered_tokens
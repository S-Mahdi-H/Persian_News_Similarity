
import stanza


# Initialize Stanza pipeline only once
_nlp = None


def get_nlp():
    """
    Load Persian Stanza pipeline.
    """

    global _nlp

    if _nlp is None:

        try:
            _nlp = stanza.Pipeline(
                lang="fa",
                processors="tokenize,mwt,pos,lemma",
                use_gpu=False,
                verbose=False,
                download_method=None
            )

        except Exception as error:
            raise RuntimeError(
                "Stanza Persian models are not available. "
                "Run the model download command first."
            ) from error

    return _nlp


def morphological_preprocess(text):
    """
    Perform tokenization, MWT expansion,
    POS tagging and lemmatization.
    """

    if not text.strip():
        return []

    nlp = get_nlp()

    document = nlp(text)

    morphological_tokens = []

    for sentence in document.sentences:

        for word in sentence.words:

            surface = word.text or ""
            lemma = word.lemma or surface

            if not surface.strip():
                continue

            morphological_tokens.append({
                "surface": surface,
                "lemma": lemma,
                "upos": word.upos
            })

    return morphological_tokens

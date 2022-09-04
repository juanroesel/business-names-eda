import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


def get_lang_detector(nlp, name):
    return LanguageDetector()

def detect_lang(nlp, text):
    # initialize language_detector component if not included in nlp pipeline
    if "language_detector" not in nlp.pipe_names:
        Language.factory("language_detector", func=get_lang_detector)
        nlp.add_pipe('language_detector', last=True)

    with nlp.select_pipes(disable=['ner', 'lemmatizer']):
        doc = nlp(text)
        return doc._.language
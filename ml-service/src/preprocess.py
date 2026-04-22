import re
import spacy

nlp = spacy.load("en_core_web_sm")
STOPWORDS = nlp.Defaults.stop_words

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\+?[\d\s\-\(\)]{7,}', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text: str) -> str:
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in STOPWORDS
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 2
    ]
    return ' '.join(tokens)

def preprocess(text: str) -> str:
    return lemmatize(clean_text(text))
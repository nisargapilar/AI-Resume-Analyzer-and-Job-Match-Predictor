import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import os

TFIDF_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'tfidf.pkl')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_tfidf():
    if os.path.exists(TFIDF_PATH):
        return joblib.load(TFIDF_PATH)
    return None

def get_tfidf_score(text1: str, text2: str, tfidf: TfidfVectorizer) -> float:
    vectors = tfidf.transform([text1, text2])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return float(score)

def get_embedding_score(text1: str, text2: str) -> float:
    embeddings = embedding_model.encode([text1, text2])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)

def get_match_score(text1: str, text2: str, tfidf: TfidfVectorizer = None) -> dict:
    embedding_score = get_embedding_score(text1, text2)

    if tfidf:
        tfidf_score = get_tfidf_score(text1, text2, tfidf)
        final_score = (tfidf_score * 0.4) + (embedding_score * 0.6)
    else:
        final_score = embedding_score

    return {
        "final_score": round(final_score * 100, 2),
        "embedding_score": round(embedding_score * 100, 2),
        "tfidf_score": round(tfidf_score * 100, 2) if tfidf else None
    }
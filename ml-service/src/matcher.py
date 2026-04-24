import joblib
import os
from src.preprocess import preprocess
from src.feature_extractor import get_match_score, load_tfidf

SKILLS = [
    "python", "javascript", "java", "sql", "machine learning", "deep learning",
    "react", "node", "docker", "kubernetes", "aws", "git", "data science",
    "tensorflow", "pytorch", "nlp", "api", "mongodb", "postgresql", "linux"
]

def extract_skills(text: str) -> list:
    text_lower = text.lower()
    return [skill for skill in SKILLS if skill in text_lower]

def match(resume_text: str, job_text: str) -> dict:
    resume_clean = preprocess(resume_text)
    job_clean = preprocess(job_text)

    tfidf = load_tfidf()
    scores = get_match_score(resume_clean, job_clean, tfidf)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))

    return {
        "match_score": scores["final_score"],
        "embedding_score": scores["embedding_score"],
        "tfidf_score": scores["tfidf_score"],
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }
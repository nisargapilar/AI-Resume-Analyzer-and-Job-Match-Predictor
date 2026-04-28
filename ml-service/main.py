from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
import joblib
import os
from src.matcher import match
from src.preprocess import preprocess

app = FastAPI(title="Resume Analyzer API")

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'saved')

def load_classifier():
    path = os.path.join(MODEL_DIR, 'classifier.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_tfidf():
    path = os.path.join(MODEL_DIR, 'tfidf.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

classifier = load_classifier()
tfidf = load_tfidf()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

class MatchRequest(BaseModel):
    resume_text: str
    job_description: str

@app.get("/health")
def health():
    return {"status": "ok", "classifier_loaded": classifier is not None}

@app.post("/analyze/text")
def analyze_text(request: MatchRequest):
    result = match(request.resume_text, request.job_description)
    
    # Predict job category
    if classifier and tfidf:
        clean = preprocess(request.resume_text)
        vector = tfidf.transform([clean])
        category = classifier.predict(vector)[0]
        result["predicted_category"] = category
    
    return result

@app.post("/analyze/pdf")
async def analyze_pdf(
    resume: UploadFile = File(...),
    job_description: str = ""
):
    if not resume.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    file_bytes = await resume.read()
    resume_text = extract_text_from_pdf(file_bytes)
    
    result = match(resume_text, job_description)
    
    if classifier and tfidf:
        clean = preprocess(resume_text)
        vector = tfidf.transform([clean])
        category = classifier.predict(vector)[0]
        result["predicted_category"] = category
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
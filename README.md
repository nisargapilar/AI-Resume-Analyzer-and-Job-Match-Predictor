This project is an AI-based Resume Analyzer and Job Match Predictor built using a full-stack architecture where the frontend is developed in React (port 5173) to allow users to upload resumes and view results, the backend is built using Node.js with Express (port 5000) which acts as the main API layer handling requests, file uploads, and communication between services, and the ML service is developed in Python using FastAPI (port 8000) where NLP and machine learning models analyze resumes using libraries like spaCy and sentence-transformers to extract skills and compute job match scores; MongoDB Atlas is used as a cloud database to store user data and results, while supporting tools like multer handle file uploads, axios manages communication between frontend and backend, and nodemon helps in development by auto-restarting the server on code changes, forming a complete pipeline from resume upload to AI-based job prediction.

# AI Resume Analyzer & Job Match Predictor

An NLP-powered resume analysis system that matches resumes to job descriptions using a hybrid TF-IDF + Sentence Embedding pipeline.

## ML Pipeline

- Text preprocessing with spaCy (lemmatization, stopword removal)
- TF-IDF vectorization trained on 2484 resumes across 24 job categories
- Semantic matching using Sentence-BERT (all-MiniLM-L6-v2)
- Resume category classification comparing 3 models

## Model Results

| Model               | Accuracy      |
| ------------------- | ------------- |
| Logistic Regression | 64.39%        |
| Linear SVM          | 68.61%        |
| **Random Forest**   | **70.02%** ✅ |

## Tech Stack

| Layer      | Technology                                           |
| ---------- | ---------------------------------------------------- |
| ML Service | Python, FastAPI, scikit-learn, sentence-transformers |
| Backend    | Node.js, Express, MongoDB Atlas                      |
| Frontend   | React (Vite)                                         |

## Project Structure

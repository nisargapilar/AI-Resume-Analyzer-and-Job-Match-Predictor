import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from src.preprocess import preprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Resume.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved')

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Categories: {df['Category'].nunique()}")
    print(df['Category'].value_counts())
    return df

def train_models(df):
    print("\nPreprocessing text...")
    df['clean'] = df['Resume_str'].apply(preprocess)

    X = df['clean']
    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    print("\nFitting TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

    # Save best model + tfidf
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} ({results[best_model_name]:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'classifier.pkl'))
    joblib.dump(best_model_name, os.path.join(MODEL_DIR, 'best_model_name.pkl'))
    print("Models saved!")

    return results

if __name__ == "__main__":
    df = load_data()
    train_models(df)
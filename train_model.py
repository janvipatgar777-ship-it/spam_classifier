"""
=============================================================================
Project Title : Spam Message Classification Using Naive Bayes Classifier
File Name     : train_model.py
Description   : This script loads the SMS spam dataset, pre-processes the text,
                extracts features using TF-IDF Vectorizer, trains a Multinomial
                Naive Bayes model, evaluates performance, and saves the trained
                model and vectorizer for the Streamlit web application.
=============================================================================
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_and_save_model():
    # -------------------------------------------------------------------------
    # Step 1: Load the Dataset
    # -------------------------------------------------------------------------
    dataset_path = os.path.join(os.path.dirname(__file__), "spam.csv")
    print(f"[INFO] Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Error: {dataset_path} does not exist. Please place spam.csv in the same folder.")

    # Load dataset using pandas
    df = pd.read_csv(dataset_path)
    print(f"[INFO] Total rows in dataset: {len(df)}")
    print("[INFO] First 5 records:")
    print(df.head())

    # Ensure required columns exist
    if "label" not in df.columns or "message" not in df.columns:
        raise ValueError("Error: CSV must contain 'label' and 'message' columns.")

    # Drop missing values if any
    df = df.dropna(subset=["label", "message"])

    # -------------------------------------------------------------------------
    # Step 2: Data Preprocessing & Encoding
    # -------------------------------------------------------------------------
    # Map labels: 'ham' -> 0 (Not Spam) and 'spam' -> 1 (Spam)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    # Display class distribution
    spam_count = df["label_num"].sum()
    ham_count = len(df) - spam_count
    print(f"\n[INFO] Class Distribution: Ham (Not Spam) = {ham_count}, Spam = {spam_count}")

    X = df["message"]
    y = df["label_num"]

    # -------------------------------------------------------------------------
    # Step 3: Train-Test Split
    # -------------------------------------------------------------------------
    # 80% data for training, 20% data for testing
    # random_state ensures reproducible results
    # stratify=y preserves the ratio of ham and spam in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # -------------------------------------------------------------------------
    # Step 4: Feature Extraction using TF-IDF Vectorizer
    # -------------------------------------------------------------------------
    # TF-IDF (Term Frequency - Inverse Document Frequency) converts text into
    # numerical feature vectors.
    # - stop_words='english': Removes common English words (e.g., 'the', 'is', 'at')
    # - lowercase=True: Converts all text to lowercase to treat 'Free' and 'free' equally
    # - max_features=3000: Keeps the top 3000 most significant words
    print("\n[INFO] Extracting features with TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=3000)

    # Fit on training data and transform both train and test
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -------------------------------------------------------------------------
    # Step 5: Model Training - Multinomial Naive Bayes
    # -------------------------------------------------------------------------
    # MultinomialNB applies Bayes' theorem with the assumption of feature independence.
    # It is particularly effective for text classification with discrete word frequencies.
    print("[INFO] Training Multinomial Naive Bayes classifier...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)

    # -------------------------------------------------------------------------
    # Step 6: Model Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("                MODEL EVALUATION RESULTS")
    print("=" * 50)

    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("  [True Ham,   False Spam]")
    print("  [False Ham,  True Spam ]")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham (Not Spam)", "Spam"]))
    print("=" * 50)

    # -------------------------------------------------------------------------
    # Step 7: Save Model & Vectorizer
    # -------------------------------------------------------------------------
    current_dir = os.path.dirname(__file__)
    model_file = os.path.join(current_dir, "model.pkl")
    vectorizer_file = os.path.join(current_dir, "vectorizer.pkl")
    metrics_file = os.path.join(current_dir, "metrics.pkl")

    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model successfully saved to: {model_file}")

    with open(vectorizer_file, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Vectorizer successfully saved to: {vectorizer_file}")

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)
    print(f"[INFO] Metrics saved to: {metrics_file}")

    print("\n[SUCCESS] Model training and export completed successfully!")

if __name__ == "__main__":
    train_and_save_model()

"""
=============================================================================
Project Title : Spam Message Classification Using Naive Bayes Classifier
File Name     : spam_classifier.py
Description   : Command-line entry point to train, test, and run the classifier.
=============================================================================
"""

import os
import sys
import pickle
from train_model import train_and_save_model

def run_cli():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")

    print("\n" + "=" * 60)
    print("   📩 Spam Message Classification Using Naive Bayes")
    print("=" * 60)

    # 1. Train model if not already saved
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("\n[INFO] Model not found. Training model now...")
        train_and_save_model()
    else:
        print("\n[INFO] Pre-trained model and vectorizer found!")

    # 2. Load model and vectorizer
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # 3. Test with sample messages
    print("\n--- Running Sample Test Predictions ---")
    test_messages = [
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call now.",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Draw! Claim now.",
        "Hey, are we still meeting for lunch at 1 PM today? Let me know!",
        "Can you please send me the class notes from yesterday's lecture? Thanks a lot!"
    ]

    for i, msg in enumerate(test_messages, 1):
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        label = "🚨 SPAM" if pred == 1 else "✅ NOT SPAM (HAM)"
        confidence = prob[pred] * 100
        print(f"\nSample {i}: \"{msg}\"")
        print(f"Result  : {label} (Confidence: {confidence:.2f}%)")

    print("\n" + "=" * 60)
    print("🚀 To launch the interactive Streamlit Web App, run:")
    print("   streamlit run app.py")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_cli()

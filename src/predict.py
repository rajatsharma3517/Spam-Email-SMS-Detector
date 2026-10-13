"""
=============================================================================
MODULE: predict.py
PROJECT: Spam Email / SMS Detection System
DESCRIPTION:
    Inference functionality for predicting whether a new, unseen Email/SMS
    message is SPAM or NOT SPAM (HAM).

INFERENCE PIPELINE:
    New Raw Message
        ↓
    NLP Preprocessing (Exact same clean_text function)
        ↓
    TF-IDF Vectorization (Using saved tfidf_vectorizer.pkl)
        ↓
    Multinomial Naive Bayes Model (Using saved spam_classifier.pkl)
        ↓
    Classification Result (SPAM / NOT SPAM + Confidence Score)
=============================================================================
"""

import os
import sys
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text


class SpamPredictor:
    """
    Spam classification engine that loads saved model artifacts and performs
    consistent inference on input messages.
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(PROJECT_ROOT, "models")
        
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "spam_classifier.pkl")
        self.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        
        self.model = None
        self.vectorizer = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads trained classifier and fitted TF-IDF vectorizer from disk."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"Model or Vectorizer artifacts missing in '{self.models_dir}'. "
                f"Please run 'python src/train_model.py' to generate artifacts."
            )
        
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def predict(self, message: str) -> dict:
        """
        Predicts whether an input message is Spam or Ham.
        
        Args:
            message (str): Raw input email or SMS message text.
            
        Returns:
            dict: {
                'raw_text': str,
                'cleaned_text': str,
                'is_spam': bool,
                'prediction_label': str ('SPAM' or 'NOT SPAM (HAM)'),
                'confidence': float (Percentage 0-100),
                'spam_probability': float (0.0 to 1.0),
                'ham_probability': float (0.0 to 1.0)
            }
        """
        if not isinstance(message, str) or not message.strip():
            return {
                "error": "Empty message provided. Please enter a valid email or SMS message.",
                "raw_text": "",
                "cleaned_text": "",
                "is_spam": False,
                "prediction_label": "UNKNOWN",
                "confidence": 0.0,
                "spam_probability": 0.0,
                "ham_probability": 0.0
            }

        # Step 1: Preprocess message using identical training logic
        cleaned_text = clean_text(message)

        # Step 2: Transform text using fitted TF-IDF vectorizer (DO NOT FIT NEW VECTORIZER)
        transformed_features = self.vectorizer.transform([cleaned_text]).toarray()

        # Step 3: Predict class label using trained model
        prediction_class = int(self.model.predict(transformed_features)[0])
        
        # Step 4: Calculate prediction probabilities with dynamic class index resolution
        probabilities = self.model.predict_proba(transformed_features)[0]
        
        # Resolve class indices dynamically from model.classes_
        classes_list = list(self.model.classes_)
        ham_idx = classes_list.index(0) if 0 in classes_list else 0
        spam_idx = classes_list.index(1) if 1 in classes_list else 1

        ham_prob = float(probabilities[ham_idx])
        spam_prob = float(probabilities[spam_idx])

        is_spam = (prediction_class == 1)
        confidence = (spam_prob if is_spam else ham_prob) * 100

        return {
            "raw_text": message,
            "cleaned_text": cleaned_text,
            "is_spam": is_spam,
            "prediction_label": "SPAM" if is_spam else "NOT SPAM (HAM)",
            "confidence": round(confidence, 2),
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4)
        }


# Convenience function for quick predictions
_default_predictor = None

def predict_message(message: str) -> dict:
    """Wrapper function to classify a single message."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = SpamPredictor()
    return _default_predictor.predict(message)


if __name__ == "__main__":
    test_messages = [
        "Congratulations! You've won a $1,000 Walmart Gift Card. Click here to claim your prize now: http://claim-prize.net",
        "Hey Mom, I'm heading over to Grandma's house right now. Let me know if you need anything from the store.",
        "URGENT: Your bank account has been locked due to suspicious activity. Call 1-800-555-0199 immediately to verify.",
        "Hi John, are we still meeting tomorrow at 3 PM for the project discussion?"
    ]

    print("=" * 75)
    print(" " * 20 + "TESTING SPAM PREDICTOR MODULE")
    print("=" * 75)

    try:
        predictor = SpamPredictor()
        for idx, msg in enumerate(test_messages, 1):
            result = predictor.predict(msg)
            print(f"\n[Test #{idx}]")
            print(f"Message   : {msg}")
            print(f"Cleaned   : {result['cleaned_text']}")
            print(f"Prediction: {result['prediction_label']} (Confidence: {result['confidence']}%)")
            print(f"Spam Prob : {result['spam_probability'] * 100:.1f}% | Ham Prob: {result['ham_probability'] * 100:.1f}%")
            print("-" * 75)
    except FileNotFoundError as e:
        print(f"[-] {e}")

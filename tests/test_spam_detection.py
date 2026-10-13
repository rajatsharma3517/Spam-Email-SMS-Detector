"""
=============================================================================
AUTOMATED TEST SUITE: test_spam_detection.py
PROJECT: Spam Email / SMS Detection System
DESCRIPTION:
    Comprehensive unit and pipeline tests verifying:
    1. Preprocessing and stopword preservation (including 'won').
    2. Model class label mapping and probability calculations.
    3. Live inference on diverse, realistic spam and legitimate messages.
    4. Model evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
    5. Robust handling of edge cases and empty inputs.
=============================================================================
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text, get_token_breakdown, STOP_WORDS
from src.predict import SpamPredictor


class TestSpamDetectionPipeline(unittest.TestCase):
    """Test suite covering preprocessing, inference, mapping, and metrics."""

    @classmethod
    def setUpClass(cls):
        """Loads model artifacts and dataset once for all test methods."""
        cls.predictor = SpamPredictor(models_dir=os.path.join(PROJECT_ROOT, "models"))
        cls.dataset_path = os.path.join(PROJECT_ROOT, "data", "spam_dataset.csv")

    # -----------------------------------------------------------------------
    # 1. Preprocessing & Stopword Tests
    # -----------------------------------------------------------------------
    def test_won_is_not_removed_as_stopword(self):
        """Verify 'won' is explicitly preserved and not discarded as a stopword."""
        self.assertNotIn(
            "won",
            STOP_WORDS,
            "'won' should not be present in STOP_WORDS set as it is a crucial spam indicator."
        )
        cleaned = clean_text("You won a prize")
        self.assertIn(
            "won",
            cleaned.split(),
            "The word 'won' must be retained in the cleaned text output."
        )

    def test_preprocessing_pipeline_normalization(self):
        """Test lowercasing, punctuation stripping, and stemming behavior."""
        raw_msg = "CONGRATULATIONS!! You've WON 1000 Cash! Call now."
        cleaned = clean_text(raw_msg)
        tokens = cleaned.split()

        self.assertIn("congratul", tokens)
        self.assertIn("won", tokens)
        self.assertIn("1000", tokens)
        self.assertIn("cash", tokens)
        self.assertIn("call", tokens)
        # Verify punctuation was removed
        for token in tokens:
            self.assertTrue(token.isalnum(), f"Token '{token}' should be alphanumeric.")

    def test_token_breakdown_structure(self):
        """Test that get_token_breakdown returns all expected intermediate steps."""
        breakdown = get_token_breakdown("Hello world 123")
        expected_keys = [
            "raw_text", "lowercased", "tokens_count",
            "cleaned_tokens", "after_stopwords", "stemmed_tokens",
            "final_cleaned_string"
        ]
        for key in expected_keys:
            self.assertIn(key, breakdown)
        self.assertEqual(breakdown["raw_text"], "Hello world 123")

    # -----------------------------------------------------------------------
    # 2. Model Class Labels and Probability Mapping
    # -----------------------------------------------------------------------
    def test_model_classes_and_probability_sum(self):
        """Verify model classes are [0, 1] and probabilities sum to 1.0."""
        model = self.predictor.model
        classes = list(model.classes_)
        self.assertEqual(classes, [0, 1], "Model classes must be [0, 1] for Ham and Spam.")

        result = self.predictor.predict("Test message for probability validation.")
        total_prob = result["ham_probability"] + result["spam_probability"]
        self.assertAlmostEqual(
            total_prob, 1.0, places=3,
            msg="Ham and Spam probabilities must sum to 1.0."
        )

    def test_prediction_label_and_confidence_consistency(self):
        """Verify label, is_spam boolean, and confidence percentage match strictly."""
        sample_messages = [
            "You won a 1000 dollar cash reward, claim now!",
            "Hey, are we having our team meeting today at 4 PM?"
        ]
        for msg in sample_messages:
            res = self.predictor.predict(msg)
            if res["is_spam"]:
                self.assertEqual(res["prediction_label"], "SPAM")
                self.assertAlmostEqual(res["confidence"], res["spam_probability"] * 100, places=1)
                self.assertGreaterEqual(res["spam_probability"], res["ham_probability"])
            else:
                self.assertEqual(res["prediction_label"], "NOT SPAM (HAM)")
                self.assertAlmostEqual(res["confidence"], res["ham_probability"] * 100, places=1)
                self.assertGreaterEqual(res["ham_probability"], res["spam_probability"])

    # -----------------------------------------------------------------------
    # 3. Target User Message: "you won a 1000 dollar rupees"
    # -----------------------------------------------------------------------
    def test_target_message_classified_as_spam(self):
        """Verify the specific user message is classified as SPAM with high confidence."""
        target_msg = "you won a 1000 dollar rupees"
        res = self.predictor.predict(target_msg)
        self.assertTrue(
            res["is_spam"],
            f"Expected '{target_msg}' to be classified as SPAM, got {res['prediction_label']} "
            f"with spam_prob={res['spam_probability']}, ham_prob={res['ham_probability']}"
        )
        self.assertEqual(res["prediction_label"], "SPAM")
        self.assertGreater(res["spam_probability"], 0.50)

    # -----------------------------------------------------------------------
    # 4. Diverse Realistic Spam Messages
    # -----------------------------------------------------------------------
    def test_diverse_spam_messages(self):
        """Verify realistic prize scams, phishing attempts, and promotional spam."""
        spam_cases = [
            "Congratulations! You have won a 1000 cash prize reward",
            "Claim your free 500 Walmart gift card now",
            "You won a guaranteed prize! Call 09061701461 to claim code KL341",
            "URGENT! Your debit card has been suspended due to suspicious unauthorized activity. Call 1-800-456-7890 immediately to reactivate.",
            "Free entry in 2 a weekly competition to win FA Cup final tkts. Text FA to 87121",
            "WINNER!! As a valued customer you have been selected to receive a 10000 cash prize reward! Call now"
        ]
        for msg in spam_cases:
            with self.subTest(msg=msg):
                res = self.predictor.predict(msg)
                self.assertTrue(
                    res["is_spam"],
                    f"Spam message failed classification: '{msg}'. Got Ham Prob={res['ham_probability']}"
                )

    # -----------------------------------------------------------------------
    # 5. Diverse Legitimate (HAM) Messages
    # -----------------------------------------------------------------------
    def test_diverse_legitimate_messages(self):
        """Verify normal personal, work, and transactional communications are HAM."""
        ham_cases = [
            "Hey, are you free for lunch tomorrow?",
            "Please find attached the updated project report for your review.",
            "Don't forget Mom's birthday dinner tonight at 7 PM.",
            "I will be late for the meeting, please start without me.",
            "Can you send me the lecture notes from today's class?",
            "Thanks for your help with the homework, I really appreciate it.",
            "The doctor confirmed your appointment for Tuesday at 3pm."
        ]
        for msg in ham_cases:
            with self.subTest(msg=msg):
                res = self.predictor.predict(msg)
                self.assertFalse(
                    res["is_spam"],
                    f"Legitimate message falsely flagged as spam: '{msg}'. Got Spam Prob={res['spam_probability']}"
                )

    # -----------------------------------------------------------------------
    # 6. Model Evaluation Metrics and Confusion Matrix
    # -----------------------------------------------------------------------
    def test_model_metrics_and_confusion_matrix(self):
        """Evaluate model against test partition and verify accuracy, precision, and zero/minimal FP."""
        self.assertTrue(os.path.exists(self.dataset_path), "Dataset file missing.")
        df = pd.read_csv(self.dataset_path)
        df.dropna(subset=["label", "message"], inplace=True)
        df.drop_duplicates(subset=["message"], inplace=True)
        df["target"] = df["label"].map({"ham": 0, "spam": 1})
        df["cleaned"] = df["message"].apply(clean_text)

        _, X_test_raw, _, y_test = train_test_split(
            df["cleaned"], df["target"],
            test_size=0.20,
            random_state=42,
            stratify=df["target"]
        )

        X_test_vec = self.predictor.vectorizer.transform(X_test_raw)
        y_pred = self.predictor.model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()

        print(f"\n[Automated Evaluation] Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, "
              f"Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%")
        print(f"[Confusion Matrix] TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        self.assertGreaterEqual(acc, 0.95, "Accuracy should be >= 95%.")
        self.assertGreaterEqual(prec, 0.98, "Precision should be >= 98% (minimizing False Positives).")
        self.assertGreaterEqual(rec, 0.70, "Recall should be >= 70%.")
        self.assertGreaterEqual(f1, 0.80, "F1-Score should be >= 80%.")
        self.assertEqual(fp, 0, "False Positives should be 0 on the benchmark test split.")

    # -----------------------------------------------------------------------
    # 7. Edge Cases and Error Handling
    # -----------------------------------------------------------------------
    def test_empty_and_whitespace_inputs(self):
        """Verify empty and whitespace inputs return structured error without crashing."""
        for invalid_input in ["", "   ", "\n\t  ", None]:
            res = self.predictor.predict(invalid_input)
            self.assertIn("error", res)
            self.assertFalse(res["is_spam"])
            self.assertEqual(res["prediction_label"], "UNKNOWN")

    def test_symbols_only_input(self):
        """Verify input with only punctuation/symbols is handled safely."""
        res = self.predictor.predict("??? !!! @@@ ### $$$")
        self.assertIn("prediction_label", res)
        self.assertIn("is_spam", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)

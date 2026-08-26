"""
=============================================================================
MODULE: train_model.py
PROJECT: Spam Email / SMS Detection System
DESCRIPTION:
    Automated training, evaluation, and artifact serialization pipeline.

WORKFLOW:
    1. Load labeled dataset (data/spam_dataset.csv).
    2. Data Cleaning: Drop duplicates and missing values.
    3. Text Preprocessing: Clean all messages using src.preprocessing.clean_text.
    4. Feature Extraction: Fit TfidfVectorizer on training text features.
    5. Train/Test Split: 80% training, 20% testing with stratification and fixed random_state.
    6. Model Training: Train Multinomial Naive Bayes classifier (MultinomialNB).
    7. Model Comparison: Also train Logistic Regression and Support Vector Machine (LinearSVC).
    8. Evaluation: Calculate Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    9. Visualization: Plot Confusion Matrix using Matplotlib and save to models/.
    10. Serialization: Save trained model, vectorizer, and metrics via Joblib / JSON.
=============================================================================
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text


def run_training_pipeline():
    print("=" * 70)
    print(" " * 15 + "SPAM DETECTION: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Define File Paths
    data_dir = os.path.join(PROJECT_ROOT, "data")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "spam_dataset.csv")

    # Step 1: Check / Load Dataset
    if not os.path.exists(dataset_path):
        print(f"[-] Dataset not found at {dataset_path}. Running download script...")
        download_script = os.path.join(data_dir, "download_dataset.py")
        if os.path.exists(download_script):
            os.system(f'python "{download_script}"')
        else:
            raise FileNotFoundError(f"Cannot find dataset at {dataset_path}")

    print(f"[+] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Standardize column names if needed
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    elif 'Category' in df.columns and 'Message' in df.columns:
        df = df[['Category', 'Message']].rename(columns={'Category': 'label', 'Message': 'message'})

    print(f"[+] Initial Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 2: Data Cleaning
    print("\n[+] Checking for missing and duplicate values...")
    initial_count = len(df)
    df.dropna(subset=['label', 'message'], inplace=True)
    df.drop_duplicates(subset=['message'], inplace=True)
    cleaned_count = len(df)
    print(f"[+] Removed {initial_count - cleaned_count} duplicate/empty records.")
    print(f"[+] Remaining unique records: {cleaned_count}")

    # Standardize labels: ham -> 0, spam -> 1
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Class distribution
    ham_count = (df['target'] == 0).sum()
    spam_count = (df['target'] == 1).sum()
    print(f"\n[+] Class Distribution:")
    print(f"    - HAM  (Legitimate): {ham_count} ({ham_count/cleaned_count*100:.2f}%)")
    print(f"    - SPAM (Malicious) : {spam_count} ({spam_count/cleaned_count*100:.2f}%)")

    # Step 3: Text Preprocessing
    print("\n[+] Applying NLP Preprocessing (Lowercasing, Tokenization, Stopwords, Stemming)...")
    df['cleaned_message'] = df['message'].apply(clean_text)

    # Step 4: Train / Test Split
    print("[+] Splitting dataset (80% Train, 20% Test) with random_state=42...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_message'],
        df['target'],
        test_size=0.20,
        random_state=42,
        stratify=df['target']
    )
    print(f"    - Training Samples: {len(X_train_raw)}")
    print(f"    - Testing Samples : {len(X_test_raw)}")

    # Step 5: TF-IDF Feature Extraction
    print("\n[+] Fitting TF-IDF Vectorizer (max_features=3000)...")
    tfidf = TfidfVectorizer(max_features=3000)
    X_train = tfidf.fit_transform(X_train_raw).toarray()
    X_test = tfidf.transform(X_test_raw).toarray()
    print(f"    - Feature Matrix Shape: {X_train.shape}")

    # Step 6: Train Primary Model (Multinomial Naive Bayes)
    print("\n[+] Training Primary Model: Multinomial Naive Bayes (MultinomialNB)...")
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)

    # Evaluate Primary Model
    y_pred_mnb = mnb.predict(X_test)
    acc_mnb = accuracy_score(y_test, y_pred_mnb)
    prec_mnb = precision_score(y_test, y_pred_mnb)
    rec_mnb = recall_score(y_test, y_pred_mnb)
    f1_mnb = f1_score(y_test, y_pred_mnb)
    cm_mnb = confusion_matrix(y_test, y_pred_mnb)

    print("\n" + "=" * 50)
    print("       PRIMARY MODEL PERFORMANCE (MultinomialNB)")
    print("=" * 50)
    print(f"Accuracy  : {acc_mnb * 100:.2f}%")
    print(f"Precision : {prec_mnb * 100:.2f}%")
    print(f"Recall    : {rec_mnb * 100:.2f}%")
    print(f"F1-Score  : {f1_mnb * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_mnb, target_names=['Ham (0)', 'Spam (1)']))

    # Step 7: Model Comparison (Logistic Regression & Linear SVM)
    print("[+] Training Comparison Models...")
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)

    # Linear Support Vector Machine
    svm = LinearSVC(random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    prec_svm = precision_score(y_test, y_pred_svm)
    rec_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)

    # Step 8: Plot Confusion Matrix using Matplotlib ONLY
    print("\n[+] Generating Confusion Matrix Plot using Matplotlib...")
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_mnb, display_labels=['Ham (Not Spam)', 'Spam'])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Confusion Matrix - Multinomial Naive Bayes', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=10, fontweight='bold')
    plt.ylabel('Actual True Label', fontsize=10, fontweight='bold')
    plt.tight_layout()

    cm_image_path = os.path.join(models_dir, "confusion_matrix.png")
    plt.savefig(cm_image_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {cm_image_path}")

    # Step 9: Save Models & Artifacts using Joblib & JSON
    classifier_path = os.path.join(models_dir, "spam_classifier.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    metrics_path = os.path.join(models_dir, "model_metrics.json")

    print("\n[+] Saving trained model and TF-IDF vectorizer using Joblib...")
    joblib.dump(mnb, classifier_path)
    joblib.dump(tfidf, vectorizer_path)

    # Save metrics JSON for UI
    metrics_data = {
        "primary_model": {
            "name": "Multinomial Naive Bayes",
            "accuracy": round(acc_mnb * 100, 2),
            "precision": round(prec_mnb * 100, 2),
            "recall": round(rec_mnb * 100, 2),
            "f1_score": round(f1_mnb * 100, 2),
            "confusion_matrix": {
                "true_negative": int(cm_mnb[0, 0]),
                "false_positive": int(cm_mnb[0, 1]),
                "false_negative": int(cm_mnb[1, 0]),
                "true_positive": int(cm_mnb[1, 1])
            }
        },
        "comparison": [
            {"model": "Multinomial Naive Bayes", "accuracy": round(acc_mnb * 100, 2), "precision": round(prec_mnb * 100, 2), "recall": round(rec_mnb * 100, 2), "f1_score": round(f1_mnb * 100, 2)},
            {"model": "Logistic Regression", "accuracy": round(acc_lr * 100, 2), "precision": round(prec_lr * 100, 2), "recall": round(rec_lr * 100, 2), "f1_score": round(f1_lr * 100, 2)},
            {"model": "Support Vector Machine", "accuracy": round(acc_svm * 100, 2), "precision": round(prec_svm * 100, 2), "recall": round(rec_svm * 100, 2), "f1_score": round(f1_svm * 100, 2)}
        ],
        "dataset_stats": {
            "total_records": cleaned_count,
            "ham_count": int(ham_count),
            "spam_count": int(spam_count),
            "vocab_size": len(tfidf.vocabulary_)
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("\n" + "=" * 70)
    print("[SUCCESS] Model training and evaluation completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_training_pipeline()

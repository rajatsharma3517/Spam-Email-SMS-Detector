# Spam Email / SMS Detection System

An end-to-end Machine Learning and Natural Language Processing (NLP) application for real-time classification of text messages and emails into **SPAM** and **LEGITIMATE (HAM)** categories.

---

## 1. System Overview

Unsolicited spam messages (phishing scams, fraudulent prize alerts, commercial spam) cause serious productivity loss and security vulnerabilities. This project implements a machine learning solution that accurately distinguishes spam messages from legitimate communications.

Using the SMS Spam Collection dataset, the system performs tokenization, stopword filtering, Porter stemming, and TF-IDF feature extraction before classifying messages with a **Multinomial Naive Bayes** classifier.

---

## 2. Key Highlights & Performance

Evaluated on an independent, unseen test dataset partition (1,034 messages):

| Metric | Score | Impact |
| :--- | :--- | :--- |
| **Accuracy** | **97.10%** | Overall correct classifications across all classes |
| **Precision** | **100.00%** | Zero false positive spam detections (Legitimate emails never flagged) |
| **Recall** | **77.10%** | Sensitivity in catching spam communications |
| **F1-Score** | **87.07%** | Harmonic balance between precision and recall |

---

## 3. Technology Stack

- **Core Language**: Python 3
- **Data Manipulation**: Pandas, NumPy
- **NLP Engine**: NLTK (Tokenization, Stopwords Removal, PorterStemmer)
- **Feature Extraction**: Scikit-Learn `TfidfVectorizer` (3,000 max features)
- **Classification Models**: Scikit-Learn `MultinomialNB`, `LogisticRegression`, `LinearSVC`
- **Visualization**: Matplotlib
- **Model Serialization**: Joblib
- **User Interface**: Streamlit (Liquid Glassmorphism styling)

---

## 4. Project Structure

```
spam_detection_project/
│
├── data/
│   ├── spam_dataset.csv          # Labeled message dataset (5,572 records)
│   └── download_dataset.py       # Dataset downloader/generator script
│
├── models/
│   ├── spam_classifier.pkl       # Trained Multinomial Naive Bayes model (Joblib)
│   ├── tfidf_vectorizer.pkl      # Fitted TF-IDF Vectorizer (Joblib)
│   ├── model_metrics.json        # Benchmark evaluation metrics (JSON)
│   └── confusion_matrix.png      # Matplotlib Confusion Matrix visualization
│
├── notebooks/
│   ├── spam_detection.ipynb      # Step-by-step Jupyter Notebook
│   └── generate_notebook.py      # Notebook generator script
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # NLP cleaning, tokenization & stemming pipeline
│   ├── train_model.py            # Training, evaluation & artifact serialization
│   └── predict.py                # Standalone inference module
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies
├── run_app.bat                   # 1-click application runner
├── train_model.bat               # 1-click model trainer
├── .gitignore                    # Git ignore configuration
└── README.md                     # Documentation
```

---

## 5. System Architecture

```
User Input Message
      ↓
NLP Preprocessing (Lowercase → Tokenize → Filter Alphanumeric → Stopwords → Stemming)
      ↓
TF-IDF Transformation (Fitted 3,000 Feature Vectorizer)
      ↓
Multinomial Naive Bayes Classifier
      ↓
Prediction Output (SPAM / NOT SPAM + Confidence Percentage)
```

---

## 6. Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train / Retrain Model
```bash
python src/train_model.py
```
*(Or double-click `train_model.bat`)*

### 3. Launch Web Application
```bash
streamlit run app.py
```
*(Or double-click `run_app.bat`)*

The application will launch in your browser at `http://localhost:8501`.

---

## 7. Sample Messages for Testing

### Spam Test Cases:
- `WINNER!! As a valued customer you have been selected to receive a 10,000 cash prize reward! Call 09061701461 now. Claim code KL341.`
- `URGENT! Your debit card has been suspended due to suspicious unauthorized activity. Call 1-800-456-7890 immediately to reactivate.`
- `Congratulations! You have been chosen for a FREE 500$ Walmart Gift Card. Click here to claim your reward: http://spamlink.cc/win`

### Legitimate (Ham) Test Cases:
- `Hi team, please remember our weekly sprint review meeting tomorrow at 10:00 AM in Conference Room B.`
- `Hey, are you coming home for dinner tonight? Mom is making your favorite pasta.`
- `Hey Rajat, could you review the machine learning project documentation when you get a chance?`

---

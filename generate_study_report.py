import os

report_text = """================================================================================
           SPAM EMAIL / SMS DETECTION SYSTEM USING MACHINE LEARNING
                   COMPLETE MASTER STUDY & INTERVIEW REPORT
================================================================================

This document is your complete, all-in-one technical guide for understanding,
explaining, and answering any interview question about your project.
Written in simple, plain English with clear step-by-step explanations.

--------------------------------------------------------------------------------
TABLE OF CONTENTS
--------------------------------------------------------------------------------
1. EXECUTIVE SUMMARY (What is this project?)
2. STEP-BY-STEP SYSTEM WORKFLOW (Zero to Hundred)
3. DATASET DETAILS & DATA CLEANING
4. NATURAL LANGUAGE PROCESSING (NLP) PREPROCESSING
5. FEATURE EXTRACTION: TF-IDF VECTORIZATION
6. MACHINE LEARNING MODEL: MULTINOMIAL NAIVE BAYES
7. EVALUATION METRICS & CONFUSION MATRIX
8. WHY HIGH PRECISION IS THE MOST IMPORTANT GOAL
9. MODEL COMPARISON (Naive Bayes vs Logistic Regression vs SVM)
10. HOW LIVE PREDICTION WORKS IN REAL TIME
11. FILE-BY-FILE CODE EXPLANATION
12. TOP 25 TECHNICAL INTERVIEW QUESTIONS & ANSWERS

================================================================================
1. EXECUTIVE SUMMARY (WHAT IS THIS PROJECT?)
================================================================================

This project is an end-to-end Machine Learning and Natural Language Processing
(NLP) system that automatically reads an incoming text message or email and
classifies it into one of two categories:
  1. SPAM: Unsolicited marketing, fraudulent lottery claims, or phishing scams.
  2. NOT SPAM (HAM): Legitimate personal, work, or transactional communications.

KEY METRICS ACHIEVED:
  - Overall Accuracy : 97.10%
  - Precision Score  : 100.00% (Zero false positive spam detections)
  - Recall Score     : 77.10%
  - F1-Score         : 87.07%
  - Technology Stack : Python, Pandas, NumPy, NLTK, Scikit-learn, Matplotlib,
                       Joblib, Streamlit

================================================================================
2. STEP-BY-STEP SYSTEM WORKFLOW (ZERO TO HUNDRED)
================================================================================

Here is the exact journey of data through our system from start to finish:

TRAINING WORKFLOW (Offline Training):
  Step 1: Load raw SMS Spam Collection CSV dataset (5,572 records).
  Step 2: Clean data by dropping null rows and removing 403 duplicate messages.
  Step 3: Encode labels: "ham" -> 0 (Legitimate), "spam" -> 1 (Spam).
  Step 4: Pass all text through NLP Preprocessing (Lowercase -> Tokenize -> 
          Filter Alphanumeric -> Remove Stopwords -> Porter Stemming).
  Step 5: Split dataset into 80% Training (4,135 samples) and 20% Testing
          (1,034 samples) using stratified sampling and fixed random_state=42.
  Step 6: Fit TF-IDF Vectorizer (max 3,000 features) ONLY on training data.
  Step 7: Transform training text into a numerical feature matrix.
  Step 8: Train Multinomial Naive Bayes classifier on training feature matrix.
  Step 9: Transform unseen test data using the fitted vectorizer and predict labels.
  Step 10: Compute Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
  Step 11: Plot Confusion Matrix using Matplotlib and save as image.
  Step 12: Serialize trained model (spam_classifier.pkl) and vectorizer 
           (tfidf_vectorizer.pkl) to disk using Joblib.

PREDICTION WORKFLOW (Real-Time Live Inference):
  Step 1: User types an email/SMS message into the Streamlit web interface.
  Step 2: The exact same clean_text() NLP function cleans the input message.
  Step 3: The pre-trained tfidf_vectorizer.pkl transforms the cleaned message
          into the exact 3,000-dimensional numerical vector format.
  Step 4: The pre-trained spam_classifier.pkl predicts whether it is 0 or 1.
  Step 5: predict_proba() calculates the exact confidence percentage for each class.
  Step 6: Streamlit UI displays the result:
          - SPAM (Rose/Red alert box) + Confidence %
          - NOT SPAM (Emerald/Green success box) + Confidence %

================================================================================
3. DATASET DETAILS & DATA CLEANING
================================================================================

1. DATASET SOURCE:
   We use the gold-standard SMS Spam Collection dataset (from the UCI Machine
   Learning Repository).

2. DATASET STRUCTURE:
   - Total Raw Rows: 5,572
   - Columns:
     * label   : Text label indicating either "ham" or "spam"
     * message : Raw text content of the email or SMS

3. CLASS DISTRIBUTION (Before Cleaning):
   - Ham (Legitimate) : 4,825 messages (86.6%)
   - Spam (Malicious) :   747 messages (13.4%)

4. DATA CLEANING PERFORMED:
   - Checked for Missing / Null values (Found 0 nulls).
   - Checked for Duplicate messages (Found 403 duplicate records).
   - Removed duplicate records to prevent data redundancy and overfitting.
   - Cleaned Dataset Size: 5,169 unique messages (4,516 Ham, 653 Spam).

5. WHY IS CLASS IMBALANCE IMPORTANT?
   Real-world communications contain vastly more legitimate emails than spam.
   Because of this imbalance (approx. 87% Ham vs 13% Spam), standard accuracy
   alone is not enough. We must evaluate Precision, Recall, and Confusion Matrix.

================================================================================
4. NATURAL LANGUAGE PROCESSING (NLP) PREPROCESSING
================================================================================

WHY IS PREPROCESSING REQUIRED?
Machine learning algorithms cannot read raw English text. They only understand
numbers. Furthermore, raw text contains noise (capital letters, punctuation,
exclamation marks, and common filler words like "is", "the", "and") that add
no predictive value.

OUR 5-STEP NLP PREPROCESSING PIPELINE:

  Step 1: Lowercasing
    - Why: Ensures words like "FREE", "Free", and "free" are treated as the
      exact same word rather than three different features.
    - Example: "FREE Cash" -> "free cash"

  Step 2: Tokenization (nltk.word_tokenize)
    - Why: Breaks continuous sentences down into individual word units (tokens).
    - Example: "win a prize now!" -> ['win', 'a', 'prize', 'now', '!']

  Step 3: Alphanumeric Filtering (word.isalnum())
    - Why: Strips out punctuation marks, commas, symbols, and formatting noise.
    - Example: ['win', 'a', 'prize', '!'] -> ['win', 'a', 'prize']

  Step 4: English Stopwords Removal (nltk.corpus.stopwords)
    - Why: Stopwords are high-frequency words (e.g., "i", "me", "is", "the",
      "at", "by", "for", "with") that appear everywhere and carry no distinct
      spam signal. Removing them drastically reduces dimensionality and focuses
      the model on meaningful keywords.
    - Example: ['you', 'have', 'won', 'a', 'free', 'gift'] -> ['won', 'free', 'gift']

  Step 5: Stemming using Porter Stemmer (nltk.stem.porter.PorterStemmer)
    - Why: Reduces words to their base root form by stripping suffixes and prefixes.
      For instance, "winner", "winning", and "wins" all reduce to "win". This
      prevents duplicate feature columns representing the same core concept.
    - Example: "congratulations you are winning" -> "congratul win"

SUMMARY OF NLP TRANSFORMATION:
  Raw Input : "Congratulations! You have WON a FREE $1,000 cash prize! Call 8888 now."
  Cleaned   : "congratul win free cash prize call 8888"

CRITICAL RULE: The exact same clean_text() function is used during training
and live prediction so that the model receives data in the exact same format.

================================================================================
5. FEATURE EXTRACTION: TF-IDF VECTORIZATION
================================================================================

WHAT IS TF-IDF?
TF-IDF stands for Term Frequency - Inverse Document Frequency.
It is a statistical measure used to evaluate how important a word is to a
document in a collection (corpus).

1. TERM FREQUENCY (TF):
   Measures how frequently a word appears in a specific message.
   Formula:
     TF(word, message) = (Count of word in message) / (Total words in message)

2. INVERSE DOCUMENT FREQUENCY (IDF):
   Measures how rare or common a word is across ALL messages in the dataset.
   Common words across all emails receive a low IDF weight, while rare words
   unique to specific messages receive a high IDF weight.
   Formula:
     IDF(word) = log_e( (Total Number of Messages) / (Number of Messages containing the word) )

3. TF-IDF SCORE:
   Formula:
     TF-IDF(word, message) = TF(word, message) * IDF(word)

WHY IS TF-IDF BETTER THAN BAG-OF-WORDS (CountVectorizer)?
  - CountVectorizer simply counts raw occurrences. If a word appears 10 times,
    its count is 10, regardless of whether it is an informative word or a common word.
  - TF-IDF penalizes generic words that appear everywhere and amplifies strong,
    distinctive spam indicator words (like "claim", "prize", "urgent", "credit").

DATA LEAKAGE PREVENTION:
  - We fit the TfidfVectorizer(max_features=3000) ONLY on training data (X_train).
  - We then transform test data (X_test) using the already fitted vectorizer.
  - Fitting on test data would be a severe error called Data Leakage.

================================================================================
6. MACHINE LEARNING MODEL: MULTINOMIAL NAIVE BAYES
================================================================================

WHAT IS NAIVE BAYES?
Naive Bayes is a supervised probabilistic classification algorithm based on
Bayes' Theorem of conditional probability:

  P(Spam | Message) = [ P(Message | Spam) * P(Spam) ] / P(Message)

Where:
  - P(Spam | Message): Posterior probability that the message is Spam given its words.
  - P(Message | Spam): Likelihood of observing these words in known spam messages.
  - P(Spam): Prior probability of spam occurring in the dataset.
  - P(Message): Total probability of the message occurring.

WHY IS IT CALLED "NAIVE"?
It is called "naive" because it makes the simplifying assumption that all words
(features) are mutually independent of each other given the class label.
Even though words in human language are correlated, this assumption works
exceptionally well in practice for text classification tasks.

WHY IS MULTINOMIAL NAIVE BAYES (MultinomialNB) BEST FOR SPAM DETECTION?
  1. Handles Discrete Count / TF-IDF Data: Designed specifically for frequency counts.
  2. Computational Efficiency: Extremely fast to train and evaluate (linear time).
  3. High Dimensionality: Works effectively even with 3,000+ sparse feature columns.
  4. Robust against Overfitting on small text datasets.

================================================================================
7. EVALUATION METRICS & CONFUSION MATRIX
================================================================================

We evaluated the model on 1,034 unseen test messages (20% test partition).

OUR EXPERIMENTAL RESULTS:
  -----------------------------------------
  Metric         Score       Meaning
  -----------------------------------------
  Accuracy       97.10%      Total correct predictions (Ham + Spam)
  Precision     100.00%      Correctly predicted spams / Total predicted spams
  Recall         77.10%      Actual spams detected / Total actual spams
  F1-Score       87.07%      Harmonic balance of Precision and Recall
  -----------------------------------------

CONFUSION MATRIX BREAKDOWN (1,034 Test Samples):
  -------------------------------------------------------------
                           Predicted HAM        Predicted SPAM
  Actual HAM (903)         TN = 903             FP = 0
  Actual SPAM (131)        FN = 30              TP = 101
  -------------------------------------------------------------

  - True Negative (TN = 903): Legitimate messages correctly classified as HAM.
  - False Positive (FP = 0)  : Legitimate messages wrongly flagged as SPAM (ZERO!).
  - False Negative (FN = 30) : Spam messages that slipped into the inbox.
  - True Positive  (TP = 101): Spam messages correctly detected and blocked.

================================================================================
8. WHY HIGH PRECISION IS THE MOST IMPORTANT GOAL
================================================================================

THIS IS THE #1 MOST POPULAR INTERVIEW QUESTION IN SPAM FILTERING:

QUESTION: "In spam detection, why is Precision more important than Recall?"

ANSWER:
  - A False Positive (FP) means a LEGITIMATE, critical email (such as an interview
    call letter, college admission notice, or bank password reset) is incorrectly
    flagged as SPAM and placed in the junk folder where the user never sees it.
  - A False Negative (FN) means an annoying spam message (like a discount coupon)
    lands in the main inbox. The user can simply delete it manually.

  Losing an important legitimate email (False Positive) causes severe damage,
  whereas seeing an extra spam message (False Negative) is merely a minor annoyance.
  Therefore, our primary engineering objective is minimizing False Positives to 0,
  which equals 100% Precision.

================================================================================
9. MODEL COMPARISON (BENCHMARK RESULTS)
================================================================================

We trained and compared three different supervised classification algorithms:

  ----------------------------------------------------------------------
  Algorithm                  Accuracy    Precision    Recall    F1-Score
  ----------------------------------------------------------------------
  Multinomial Naive Bayes     97.10%      100.00%     77.10%     87.07%  <-- PRIMARY
  Logistic Regression         96.03%       96.88%     70.99%     81.94%
  Support Vector Machine      98.07%       97.44%     87.02%     91.94%
  ----------------------------------------------------------------------

WHY MULTINOMIAL NAIVE BAYES WAS CHOSEN AS THE PRIMARY MODEL:
  - It achieved a perfect 100.00% Precision on test data (0 false positives).
  - It trains in milliseconds and requires minimal memory footprint.
  - It provides exact posterior probabilities via predict_proba().

================================================================================
10. HOW LIVE PREDICTION WORKS IN REAL TIME
================================================================================

When you enter a message in the Streamlit web application:

1. Input:
   "URGENT: Your debit card is blocked. Call 1-800-456-7890 to verify account."

2. Preprocessing (`src/preprocessing.py`):
   - Lowercase: "urgent: your debit card is blocked. call 1-800-456-7890 to verify account."
   - Tokenize: ['urgent', 'your', 'debit', 'card', 'is', 'blocked', 'call', '1-800-456-7890', 'to', 'verify', 'account']
   - Alphanumeric: ['urgent', 'your', 'debit', 'card', 'is', 'blocked', 'call', 'to', 'verify', 'account']
   - Remove Stopwords: ['urgent', 'debit', 'card', 'blocked', 'call', 'verify', 'account']
   - Stemming: ['urgent', 'debit', 'card', 'block', 'call', 'verifi', 'account']
   - Final string: "urgent debit card block call verifi account"

3. TF-IDF Transformation:
   - Passed into `tfidf_vectorizer.pkl`.
   - Generates a 1 x 3,000 sparse numerical vector where indices corresponding
     to 'urgent', 'debit', 'card', 'block', 'call', 'verifi', 'account' receive
     high mathematical weights.

4. Naive Bayes Classification:
   - Vector passed into `spam_classifier.pkl`.
   - Computes:
       P(Spam | Vector) = 94.2%
       P(Ham  | Vector) =  5.8%
   - Predicted Class = 1 (SPAM).

5. UI Output:
   - Renders 🚨 SPAM DETECTED banner with 94.2% confidence score.

================================================================================
11. FILE-BY-FILE CODE EXPLANATION
================================================================================

1. `data/spam_dataset.csv`:
   - Contains 5,572 labeled text messages used for training and testing.

2. `src/preprocessing.py`:
   - Contains `clean_text(text: str) -> str` and `get_token_breakdown(text: str) -> dict`.
   - Handles automatic NLTK resource downloading ('punkt', 'stopwords').
   - Contains the reusable 5-step NLP pipeline.

3. `src/train_model.py`:
   - End-to-end training script.
   - Loads CSV, removes duplicates, applies preprocessing, splits data (80/20).
   - Fits TF-IDF vectorizer and trains MultinomialNB, LogisticRegression, and LinearSVC.
   - Saves `spam_classifier.pkl`, `tfidf_vectorizer.pkl`, and `model_metrics.json`.
   - Plots and saves `confusion_matrix.png` using Matplotlib.

4. `src/predict.py`:
   - Inference module containing `SpamPredictor` class.
   - Loads saved `.pkl` files and provides `.predict(message)` method returning
     label, confidence percentage, and class probabilities.

5. `app.py`:
   - Interactive Streamlit web interface styled with Light Liquid-Glassmorphism.
   - Features 2 main tabs:
     * Tab 1: Live Message Predictor (text input, classify/clear buttons, breakdown expander)
     * Tab 2: Model Performance & Metrics (score cards, confusion matrix, comparison table)
   - Clean sidebar with architecture pipeline and preset Quick Test Samples.

6. `run_app.bat` & `train_model.bat`:
   - 1-click execution batch scripts for Windows.

================================================================================
12. TOP 25 TECHNICAL INTERVIEW QUESTIONS & ANSWERS
================================================================================

Q1: What is the objective of this project?
A1: To build an automated NLP and Machine Learning classification system that
    accurately categorizes text messages and emails as either SPAM or HAM (legitimate).

Q2: What is Natural Language Processing (NLP)?
A2: NLP is a subfield of Artificial Intelligence that allows computer algorithms
    to analyze, process, and extract meaningful patterns from human language text.

Q3: Why can't we feed raw text directly to machine learning models?
A3: Machine learning algorithms rely on linear algebra and mathematical matrix
    computations. They cannot process characters, strings, or sentences directly.
    Text must first be cleaned and converted into numerical feature vectors.

Q4: What is Tokenization?
A4: Tokenization is the process of splitting continuous sentences or paragraphs
    into discrete semantic units called tokens (individual words).

Q5: What are Stop Words and why do we remove them?
A5: Stop words are high-frequency words in a language (like "is", "the", "at",
    "on", "by") that occur everywhere and contain little domain information.
    Removing them reduces dimensionality and prevents noise.

Q6: What is the difference between Stemming and Lemmatization?
A6: Stemming uses heuristic rule-based affix stripping to reduce words to a base
    root (e.g., "studying" -> "studi"), which might not be a dictionary word.
    Lemmatization uses vocabulary and morphological analysis to return the
    grammatically correct base lemma (e.g., "better" -> "good"). We used
    Porter Stemmer for its speed and effectiveness on short text.

Q7: What is TF-IDF?
A7: TF-IDF (Term Frequency - Inverse Document Frequency) is a numerical statistic
    that quantifies word importance. It multiplies how often a word appears in a
    document (TF) by how rare the word is across the whole dataset (IDF).

Q8: What is the mathematical formula for TF-IDF?
A8: TF(t, d) = (count of t in d) / (total words in d)
    IDF(t) = log( (Total documents N) / (Documents containing t) )
    TF-IDF = TF(t, d) * IDF(t)

Q9: What is the difference between Bag of Words and TF-IDF?
A9: Bag of Words (CountVectorizer) only records raw frequencies, giving high
    weight to frequent words regardless of their meaning. TF-IDF balances
    frequency with rarity, penalizing common words and highlighting informative
    distinctive keywords.

Q10: What is Bayes' Theorem?
A10: Bayes' Theorem calculates conditional probability:
     P(A|B) = [P(B|A) * P(A)] / P(B)
     It determines the probability of class A occurring given evidence B.

Q11: Why is Multinomial Naive Bayes called "Naive"?
A11: Because it assumes all features (word occurrences) are completely independent
     of each other given the class, ignoring word order and semantic context.

Q12: Why is Multinomial Naive Bayes well-suited for text classification?
A12: It works natively with discrete frequency counts and sparse high-dimensional
     matrices, is computationally fast, and is resilient to irrelevant features.

Q13: What is the Train-Test Split ratio used in your project?
A13: An 80% training set (4,135 samples) and a 20% test set (1,034 samples),
     stratified to preserve class distribution ratios, using random_state=42.

Q14: What is Data Leakage and how did you prevent it?
A14: Data Leakage occurs when information from test data contaminates model
     training. We prevented this by fitting the TF-IDF vectorizer ONLY on
     training data (X_train) and merely transforming X_test and live messages.

Q15: Why must we use the exact same TF-IDF vectorizer during prediction?
A15: The vectorizer assigns specific vocabulary words to fixed matrix column indices.
     Fitting a new vectorizer during prediction would create index mismatches
     and corrupt all model inputs.

Q16: What is Accuracy?
A16: Accuracy = (True Positives + True Negatives) / (Total Predictions).
     It measures overall correctness. Our model achieved 97.10% accuracy.

Q17: What is Precision?
A17: Precision = True Positives / (True Positives + False Positives).
     It measures what fraction of messages predicted as spam were truly spam.
     Our model achieved 100.00% precision.

Q18: What is Recall (Sensitivity)?
A18: Recall = True Positives / (True Positives + False Negatives).
     It measures what fraction of all actual spam messages were detected.
     Our model achieved 77.10% recall.

Q19: What is F1-Score?
A19: F1-Score = 2 * (Precision * Recall) / (Precision + Recall).
     It is the harmonic mean of Precision and Recall. Our model achieved 87.07%.

Q20: Why is Precision more important than Recall in spam detection?
A20: A False Positive misclassifies a critical legitimate email as spam, causing
     the user to miss it. A False Negative merely leaves a spam email in the
     inbox to be deleted. Minimizing False Positives (maximizing Precision) is
     the primary engineering objective.

Q21: What is a Confusion Matrix?
A21: A 2x2 matrix comparing actual ground-truth labels against model predictions,
     displaying True Negatives, False Positives, False Negatives, and True Positives.

Q22: How many False Positives did your model produce on test data?
A22: Zero (FP = 0 out of 1,034 test samples), confirming 100% precision.

Q23: How are the trained model and vectorizer saved to disk?
A23: Using Joblib (`joblib.dump`), which efficiently serializes Python scikit-learn
     objects containing large NumPy arrays into `.pkl` binary files.

Q24: What is the role of Streamlit in this project?
A24: Streamlit provides a clean, reactive Python-based web user interface for
     live user message input, real-time prediction display, and interactive metrics.

Q25: If you had more time, how would you improve this system?
A25: 1. Incorporate character-level n-grams to catch intentionally misspelled
        spam words (like "w!nner" or "c@sh").
     2. Implement sender domain reputation scoring and metadata analysis.
     3. Add active user feedback logging to continuously retrain the classifier.

================================================================================
                      END OF MASTER STUDY REPORT
================================================================================
"""

# Write to TXT file for Notepad
txt_path = r"D:\spam_detection_project\PROJECT_STUDY_REPORT.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"Master Study Report (TXT) generated at: {txt_path}")

# Write to Markdown file for VS Code
md_path = r"D:\spam_detection_project\PROJECT_STUDY_REPORT.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"Master Study Report (MD) generated at: {md_path}")

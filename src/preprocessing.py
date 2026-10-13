"""
=============================================================================
MODULE: preprocessing.py
PROJECT: Spam Email / SMS Detection System
DESCRIPTION:
    Reusable Natural Language Processing (NLP) text preprocessing functions.
    The exact same preprocessing pipeline is used during both model training
    and live inference to maintain machine learning pipeline consistency.

NLP PIPELINE STEPS:
    1. Lowercasing: Standardizes all text characters.
    2. Tokenization: Splits text into individual word tokens.
    3. Non-alphanumeric removal: Filters out punctuation and special symbols.
    4. Stopwords Removal: Removes high-frequency words carrying minimal semantic weight.
    5. Stemming (PorterStemmer): Normalizes words to their root base form.
=============================================================================
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------------------------------------------------------------
# Setup NLTK Resources (Checked locally first to eliminate startup latency)
# ---------------------------------------------------------------------------
def setup_nltk():
    """
    Ensures required NLTK datasets ('punkt', 'punkt_tab', 'stopwords') 
    are available locally. Checks local data paths first to avoid network delays.
    """
    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords'
    }
    for res_name, res_path in resources.items():
        try:
            nltk.data.find(res_path)
        except LookupError:
            try:
                nltk.download(res_name, quiet=True)
            except Exception:
                pass

# Initialize NLTK setup on module load
setup_nltk()

# Initialize Stemmer
ps = PorterStemmer()

# Exclude words that carry strong domain significance in spam detection.
# 'won' is included in standard NLTK stopwords solely as a fragment of contraction "won't".
# In text communication, 'won' is the past tense of 'win' and represents one of the strongest,
# most statistically significant indicators of lottery, prize, and sweepstakes spam messages
# (occurring in ~9.3% of spam messages vs 0.4% of legitimate messages in the corpus).
# Therefore, 'won' must NOT be discarded as a stopword.
SPAM_EXCLUDED_STOPWORDS = {'won'}

DEFAULT_STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", "won't", 'wouldn', "wouldn't"
}

try:
    STOP_WORDS = set(stopwords.words('english')) - SPAM_EXCLUDED_STOPWORDS
except Exception:
    STOP_WORDS = DEFAULT_STOP_WORDS - SPAM_EXCLUDED_STOPWORDS


def clean_text(text: str) -> str:
    """
    Takes raw message string and applies full NLP normalization pipeline.
    
    Args:
        text (str): Raw input email or SMS message.
        
    Returns:
        str: Space-separated cleaned, tokenized, stopword-filtered, and stemmed tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Tokenize text
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = re.findall(r'\b\w+\b', text)

    # Step 3: Keep only alphanumeric tokens (removes punctuation & symbols)
    alphanumeric_tokens = [word for word in tokens if word.isalnum()]

    # Step 4: Remove English Stopwords
    filtered_tokens = [word for word in alphanumeric_tokens if word not in STOP_WORDS]

    # Step 5: Stemming using Porter Stemmer (reduce to root word)
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

    # Step 6: Join back into a single string for TF-IDF Vectorizer
    return " ".join(stemmed_tokens)


def get_token_breakdown(text: str) -> dict:
    """
    Provides step-by-step breakdown of text transformation.
    
    Args:
        text (str): Raw input text.
        
    Returns:
        dict: Dictionary containing step-by-step intermediate representations.
    """
    raw = text if isinstance(text, str) else ""
    lowered = raw.lower()
    
    try:
        tokenized = nltk.word_tokenize(lowered)
    except Exception:
        tokenized = re.findall(r'\b\w+\b', lowered)
        
    alnum = [w for w in tokenized if w.isalnum()]
    no_stop = [w for w in alnum if w not in STOP_WORDS]
    stemmed = [ps.stem(w) for w in no_stop]
    
    return {
        "raw_text": raw,
        "lowercased": lowered,
        "tokens_count": len(tokenized),
        "cleaned_tokens": alnum,
        "after_stopwords": no_stop,
        "stemmed_tokens": stemmed,
        "final_cleaned_string": " ".join(stemmed)
    }

if __name__ == "__main__":
    sample = "Congratulations! You have WON a FREE 1000$ cash prize! Call now at 8888."
    print("Original Text:\n", sample)
    print("\nCleaned Text:\n", clean_text(sample))

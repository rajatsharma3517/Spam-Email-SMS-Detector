"""
=============================================================================
APPLICATION: app.py
PROJECT: Spam Email / SMS Detection System
DESCRIPTION:
    A lightweight, modern web interface with a clean Light Liquid-Glassmorphism
    visual design for real-time classification of emails and SMS messages.
=============================================================================
"""

import os
import sys
import json
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Email / SMS Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text, get_token_breakdown
from src.predict import SpamPredictor

# LIGHT LIQUID GLASSMORPHISM CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Typography & App Base */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Default Deploy Button & Three-Dot Menu */
    [data-testid="stAppDeployButton"], 
    .stAppDeployButton,
    #MainMenu,
    [data-testid="stMainMenu"],
    header[data-testid="stHeader"] button[aria-label="Manage app"],
    header[data-testid="stHeader"] button[aria-label="Settings"],
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Remove Large Top White Space / Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1150px !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2rem !important;
    }

    /* Light Liquid Ambient Background */
    .stApp {
        background: 
            radial-gradient(circle at 5% 5%, rgba(99, 102, 241, 0.05), transparent 30%),
            radial-gradient(circle at 95% 10%, rgba(236, 72, 153, 0.04), transparent 25%),
            radial-gradient(circle at 50% 90%, rgba(59, 130, 246, 0.04), transparent 35%),
            #f8fafc;
        color: #1e293b;
    }

    /* Light Glass Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }

    /* Main Title Styling */
    h1 {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.25rem !important;
    }

    /* Subtitle */
    .subtitle-text {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.25rem;
    }

    /* Clean Light Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(241, 245, 249, 0.8);
        backdrop-filter: blur(8px);
        padding: 4px;
        border-radius: 10px;
        border: 1px solid rgba(226, 232, 240, 0.9);
        margin-bottom: 1.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 8px;
        padding: 0 16px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
        background: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #4f46e5 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }

    /* Textarea Light Glass Styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        font-size: 0.95rem !important;
        padding: 12px 14px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }

    /* Buttons */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        border: 1px solid #4338ca !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
    }

    .stButton button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #475569 !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
    }

    .stButton button[kind="secondary"]:hover {
        background: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    /* Metric Cards - Light Glass */
    .light-metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .light-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }

    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin: 2px 0;
    }

    .metric-lbl {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }

    .metric-sub {
        font-size: 0.75rem;
        color: #94a3b8;
    }

    /* Light Result Cards */
    .light-result-spam {
        background: rgba(254, 242, 242, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #fecaca;
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 18px 20px;
        margin-top: 14px;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.08);
    }

    .light-result-ham {
        background: rgba(240, 253, 244, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 18px 20px;
        margin-top: 14px;
        box-shadow: 0 2px 8px rgba(34, 197, 94, 0.08);
    }

    /* Light Glass Section Box */
    .light-glass-box {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
    }

    /* Clean Footer */
    .light-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 36px;
        padding-top: 16px;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Data & Model Loader
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_metrics():
    models_dir = os.path.join(PROJECT_ROOT, "models")
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    cm_image_path = os.path.join(models_dir, "confusion_matrix.png")
    
    predictor = None
    metrics_data = None
    error_msg = None

    try:
        predictor = SpamPredictor(models_dir=models_dir)
    except Exception as e:
        error_msg = str(e)

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
        except Exception:
            metrics_data = None

    return predictor, metrics_data, cm_image_path, error_msg


predictor, metrics_data, cm_image_path, load_error = load_model_and_metrics()

# ---------------------------------------------------------------------------
# SIDEBAR: Overview & Quick Test Samples
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Spam Shield")
st.sidebar.caption("Machine Learning Detection System")

st.sidebar.markdown("---")
st.sidebar.subheader("Architecture Pipeline")
st.sidebar.markdown(
    """
    - **Input**: Email or SMS text
    - **Preprocessing**: Tokenization & Stemming (NLTK)
    - **Features**: TF-IDF (3,000 features)
    - **Classifier**: Multinomial Naive Bayes
    - **Output**: Real-time Spam/Ham probability
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Test Samples")
sample_choice = st.sidebar.selectbox(
    "Choose a preset message to test:",
    [
        "Select a sample...",
        "[SPAM] Free Lottery / Cash Reward",
        "[SPAM] Urgent Bank Security Alert",
        "[SPAM] Gift Card & Coupon Winner",
        "[HAM] Team Meeting Reminder",
        "[HAM] Personal Family Message",
        "[HAM] Project Discussion Request"
    ]
)

sample_texts = {
    "[SPAM] Free Lottery / Cash Reward": "WINNER!! As a valued customer you have been selected to receive a 10,000 cash prize reward! Call 09061701461 now. Claim code KL341.",
    "[SPAM] Urgent Bank Security Alert": "URGENT! Your debit card has been suspended due to suspicious unauthorized activity. Call 1-800-456-7890 immediately to reactivate.",
    "[SPAM] Gift Card & Coupon Winner": "Congratulations! You have been chosen for a FREE 500$ Walmart Gift Card. Click here to claim your reward: http://spamlink.cc/win",
    "[HAM] Team Meeting Reminder": "Hi team, please remember our weekly sprint review meeting tomorrow at 10:00 AM in Conference Room B.",
    "[HAM] Personal Family Message": "Hey, are you coming home for dinner tonight? Mom is making your favorite pasta.",
    "[HAM] Project Discussion Request": "Hey Rajat, could you review the machine learning project documentation when you get a chance?"
}

default_text = ""
if sample_choice in sample_texts:
    default_text = sample_texts[sample_choice]

st.sidebar.markdown("---")
st.sidebar.caption("Engine: Python • Scikit-learn • NLTK")

# ---------------------------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------------------------
st.title("📧 Spam Email / SMS Detection System")
st.markdown(
    '<div class="subtitle-text">Analyze email and SMS messages using machine learning to detect potential spam.</div>',
    unsafe_allow_html=True
)

if load_error:
    st.error(
        f"⚠️ **Model Artifacts Not Found**: {load_error}\n\n"
        "Please run `python src/train_model.py` to train and save the model artifacts."
    )
    st.stop()

# ---------------------------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "🔍 Live Message Predictor",
    "📊 Model Performance & Metrics"
])

# ===========================================================================
# TAB 1: LIVE MESSAGE PREDICTOR
# ===========================================================================
with tab1:
    st.subheader("Message Classification")
    
    user_input = st.text_area(
        label="Enter Email / SMS Message Text:",
        value=default_text,
        height=130,
        placeholder="Type or paste an email or SMS message here to classify..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("🚀 Classify Message", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Clear Input", type="secondary"):
            user_input = ""
            st.rerun()

    if predict_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter some text before clicking Classify.")
        else:
            with st.spinner("Analyzing message through NLP pipeline..."):
                result = predictor.predict(user_input)
                breakdown = get_token_breakdown(user_input)

            # Prediction Result Display
            if result["is_spam"]:
                st.markdown(
                    f"""
                    <div class="light-result-spam">
                        <div style="font-size: 1.15rem; font-weight: 700; color: #b91c1c; margin-bottom: 4px;">
                            🚨 Prediction: SPAM MESSAGE
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #991b1b; margin-bottom: 6px;">
                            Confidence Score: {result['confidence']:.2f}%
                        </div>
                        <div style="font-size: 0.88rem; color: #7f1d1d;">
                            This message exhibits characteristics commonly associated with unsolicited promotions or phishing attempts.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="light-result-ham">
                        <div style="font-size: 1.15rem; font-weight: 700; color: #15803d; margin-bottom: 4px;">
                            ✅ Prediction: NOT SPAM (HAM)
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #166534; margin-bottom: 6px;">
                            Confidence Score: {result['confidence']:.2f}%
                        </div>
                        <div style="font-size: 0.88rem; color: #14532d;">
                            This message appears to be standard, legitimate communication.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Probability Distribution
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("#### Probability Distribution")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.metric("Legitimate (Ham) Probability", f"{result['ham_probability'] * 100:.2f}%")
                st.progress(min(max(result['ham_probability'], 0.0), 1.0))
            with p_col2:
                st.metric("Spam Probability", f"{result['spam_probability'] * 100:.2f}%")
                st.progress(min(max(result['spam_probability'], 0.0), 1.0))

            # NLP Breakdown Expander
            with st.expander("🔍 Inspect NLP Preprocessing Steps"):
                st.markdown("**1. Raw Input Text:**")
                st.code(breakdown["raw_text"])

                st.markdown("**2. Filtered Alphanumeric Tokens:**")
                st.write(breakdown["cleaned_tokens"])

                st.markdown("**3. After Stopwords Removal:**")
                st.write(breakdown["after_stopwords"])

                st.markdown("**4. After Stemming (Porter Stemmer):**")
                st.write(breakdown["stemmed_tokens"])

                st.markdown("**5. Final Cleaned String fed to TF-IDF:**")
                st.info(breakdown["final_cleaned_string"] if breakdown["final_cleaned_string"] else "[No keywords remaining]")


# ===========================================================================
# TAB 2: MODEL PERFORMANCE & METRICS
# ===========================================================================
with tab2:
    st.subheader("Model Evaluation & Experimental Results")
    st.markdown("Evaluated on an unseen test dataset split (20% stratified test partition).")

    if metrics_data:
        pm = metrics_data.get("primary_model", {})
        
        # 4 Light Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(
                f"""
                <div class="light-metric-card">
                    <div class="metric-lbl">Accuracy</div>
                    <div class="metric-val">{pm.get('accuracy', 0.0)}%</div>
                    <div class="metric-sub">Total correct predictions</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with m_col2:
            st.markdown(
                f"""
                <div class="light-metric-card">
                    <div class="metric-lbl">Precision</div>
                    <div class="metric-val" style="color: #059669;">{pm.get('precision', 0.0)}%</div>
                    <div class="metric-sub">Zero false positive spam tags</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with m_col3:
            st.markdown(
                f"""
                <div class="light-metric-card">
                    <div class="metric-lbl">Recall</div>
                    <div class="metric-val">{pm.get('recall', 0.0)}%</div>
                    <div class="metric-sub">Spam detection sensitivity</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with m_col4:
            st.markdown(
                f"""
                <div class="light-metric-card">
                    <div class="metric-lbl">F1-Score</div>
                    <div class="metric-val">{pm.get('f1_score', 0.0)}%</div>
                    <div class="metric-sub">Harmonic mean balance</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Confusion Matrix Section
        cm_col1, cm_col2 = st.columns([1, 1])
        
        with cm_col1:
            st.markdown("### 📉 Confusion Matrix (Matplotlib)")
            if os.path.exists(cm_image_path):
                img = Image.open(cm_image_path)
                st.image(img, caption="Confusion Matrix on Test Dataset", use_container_width=True)
            else:
                st.info("Confusion matrix image not found. Run train_model.py to generate.")
            
        with cm_col2:
            st.markdown("### 🧩 Confusion Matrix Breakdown")
            cm = pm.get("confusion_matrix", {})
            tn = cm.get("true_negative", 0)
            fp = cm.get("false_positive", 0)
            fn = cm.get("false_negative", 0)
            tp = cm.get("true_positive", 0)

            st.markdown(
                f"""
                <div class="light-glass-box">
                    <div style="line-height: 1.8; color: #334155; font-size: 0.92rem;">
                        • <b>True Negatives (TN):</b> <code>{tn}</code> (Legitimate messages correctly classified as Ham)<br>
                        • <b>False Positives (FP):</b> <code style="color: #059669; font-weight: bold;">{fp}</code> (Legitimate messages wrongly marked as Spam)<br>
                        • <b>False Negatives (FN):</b> <code>{fn}</code> (Spam messages that slipped through)<br>
                        • <b>True Positives (TP):</b> <code>{tp}</code> (Spam messages correctly blocked)
                    </div>
                    <div style="margin-top: 14px; padding: 10px 12px; background: rgba(79, 70, 229, 0.06); border-left: 3px solid #4f46e5; border-radius: 4px; font-size: 0.84rem; color: #334155;">
                        <b>Key Engineering Insight:</b> In spam filtering, <b>Precision is prioritized over Recall</b>. Achieving zero false positives ensures legitimate emails are never erroneously lost to spam folders.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Model Comparison Table
        st.markdown("### ⚖️ Algorithm Comparison")
        st.markdown("Performance comparison across supervised learning models:")
        
        comp = metrics_data.get("comparison", [])
        if comp:
            st.table(comp)

        ds = metrics_data.get("dataset_stats", {})
        if ds:
            st.caption(
                f"Dataset Records: {ds.get('total_records', 0)} | "
                f"Ham Count: {ds.get('ham_count', 0)} | "
                f"Spam Count: {ds.get('spam_count', 0)} | "
                f"TF-IDF Vocabulary: {ds.get('vocab_size', 0)} features"
            )
    else:
        st.info("Run `python src/train_model.py` to generate complete evaluation metrics.")

# ---------------------------------------------------------------------------
# CLEAN FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="light-footer">
        Spam Email / SMS Detection System • Machine Learning & Natural Language Processing
    </div>
    """,
    unsafe_allow_html=True
)

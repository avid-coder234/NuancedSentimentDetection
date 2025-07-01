import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import nltk
import sklearn
# === Download necessary NLTK data ===
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, sentiwordnet as swn, wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from sklearn.metrics import classification_report

# === Version Logging ===
st.sidebar.info(f"🔍 scikit-learn version: {sklearn.__version__}")
st.sidebar.info(f"📦 NLTK version: {nltk.__version__}")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# === Check for model and vectorizer files ===
MODEL_PATH = 'model_zero.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error(f"""
    🚫 Required files not found:
    - {'✅' if os.path.exists(MODEL_PATH) else '❌ model_zero.pkl'}
    - {'✅' if os.path.exists(VECTORIZER_PATH) else '❌ tfidf_vectorizer.pkl'}

    Please ensure both files are present in the same directory.
    """)
    st.stop()

# === Load model and vectorizer ===
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    st.error(f"⚠️ Error loading model/vectorizer: {str(e)}")
    st.stop()


# Initialize stop words and sentiment analyzer
stop_words = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

# Custom CSS for better styling
st.set_page_config(
    page_title="Nuanced Sentiment Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 2px solid;
    }
    
    .sarcastic {
        border-color: #ff6b6b;
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
        color: white;
    }
    
    .not-sarcastic {
        border-color: #51cf66;
        background: linear-gradient(135deg, #51cf66 0%, #69db7c 100%);
        color: white;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .feature-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    .emoji-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def get_sentiment_scores(word):
    synsets = wn.synsets(word)
    if not synsets:
        return 0
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return swn_synset.pos_score() - swn_synset.neg_score()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(tokens)

def extract_pragmatic_features(text):
    features = {
        'capitalization': any(char.isupper() for char in text),
        'num_exclamation': text.count('!'),
        'num_question': text.count('?'),
        'num_punctuation': sum([1 for char in text if char in '.,;:']),
        'num_emoticons': sum([1 for word in text.split() if word in [':)', ':(', ':-)', ':-(', ':D', 'XD', ':P', ':o', ':-D', ':-P', ':-o']]),
        'num_lol_haha': text.lower().count('lol') + text.lower().count('haha') + text.lower().count('hehe'),
        'sentiment_score': sid.polarity_scores(text)['compound']
    }
    return list(features.values())

def extract_explicit_incongruity_features(text):
    tokens = word_tokenize(text)
    positive_words = [word for word in tokens if get_sentiment_scores(word) > 0]
    negative_words = [word for word in tokens if get_sentiment_scores(word) < 0]
    
    pos_count = len(positive_words)
    neg_count = len(negative_words)
    sentiment_incongruities = abs(pos_count - neg_count)
    
    largest_pos_seq = max([len(seq) for seq in re.findall(r'(?:\b(?:' + '|'.join(positive_words) + r')\b\s*)+', text)], default=0)
    largest_neg_seq = max([len(seq) for seq in re.findall(r'(?:\b(?:' + '|'.join(negative_words) + r')\b\s*)+', text)], default=0)

    return [
        sentiment_incongruities,
        largest_pos_seq,
        largest_neg_seq,
        pos_count,
        neg_count
    ]

def extract_implicit_incongruity_features(text):
    verb_phrases = len([word for word in word_tokenize(text) if any(syn.pos() == 'v' for syn in wn.synsets(word))])
    noun_phrases = len([word for word in word_tokenize(text) if any(syn.pos() == 'n' for syn in wn.synsets(word))])
    return [verb_phrases, noun_phrases]

def preprocess_and_extract_features(texts):
    clean_texts = []
    features = []
    for text in texts:
        clean_texts.append(preprocess_text(text))
        pragmatic_features = extract_pragmatic_features(text)
        explicit_features = extract_explicit_incongruity_features(text)
        implicit_features = extract_implicit_incongruity_features(text)
        features.append(pragmatic_features + explicit_features + implicit_features)
    return clean_texts, np.array(features)

def predict_sarcasm(texts):
    clean_texts, additional_features = preprocess_and_extract_features(texts)
    X_lexical = vectorizer.transform(clean_texts).toarray()
    X_combined = np.hstack((X_lexical, additional_features))
    return model.predict(X_combined)

# Load training classification report
training_report = {
    'precision': [0.99, 0.08],
    'recall': [0.70, 0.86],
    'f1-score': [0.82, 0.15],
    'support': [2611, 83]
}

accuracy = 0.71
macro_avg = [0.54, 0.78, 0.49]
weighted_avg = [0.97, 0.71, 0.80]

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧠 Nuanced Sentiment Detection</h1>
    <p>Advanced AI-powered sarcasm detection using pragmatic and incongruity features</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with improved styling
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    option = st.selectbox(
        "Choose your option:",
        ("📁 Upload CSV File", "✍️ Input Text Query", "📊 View Evaluation Metrics"),
        format_func=lambda x: x.split(" ", 1)[1]
    )
    
    # Model metrics in sidebar
    st.markdown("### 📈 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("F1 Score", f"{training_report['f1-score'][0]:.2f}")

# Main content based on selection
if "Upload CSV File" in option:
    st.markdown("## 📁 Upload CSV File")
    st.markdown("Upload a CSV file containing text data for batch sarcasm detection.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="File should contain a 'body' column with text data"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing your data..."):
            input_data = pd.read_csv(uploaded_file)
            
            # Display input data
            st.markdown("### 📋 Input Data Preview")
            st.dataframe(input_data.head(), use_container_width=True)
            
            # Prediction
            input_texts = input_data['body'].values
            predictions = predict_sarcasm(input_texts)
            input_data['predicted_sarcasm_tag'] = predictions
            
            # Results summary
            sarcastic_count = (predictions == 'yes').sum()
            total_count = len(predictions)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Texts", total_count)
            with col2:
                st.metric("Sarcastic", sarcastic_count, f"{sarcastic_count/total_count:.1%}")
            with col3:
                st.metric("Not Sarcastic", total_count - sarcastic_count, f"{(total_count-sarcastic_count)/total_count:.1%}")
            
            # Display results with improved styling
            st.markdown("### 🎯 Prediction Results")
            
            # Filter options
            filter_option = st.selectbox("Filter results:", ["All", "Sarcastic", "Not Sarcastic"])
            
            if filter_option == "Sarcastic":
                filtered_data = input_data[input_data['predicted_sarcasm_tag'] == 'yes']
            elif filter_option == "Not Sarcastic":
                filtered_data = input_data[input_data['predicted_sarcasm_tag'] == 'no']
            else:
                filtered_data = input_data
            
            # Display filtered results
            for idx, row in filtered_data.iterrows():
                prediction_class = "sarcastic" if row['predicted_sarcasm_tag'] == 'yes' else "not-sarcastic"
                icon = "😏" if row['predicted_sarcasm_tag'] == 'yes' else "😊"
                
                st.markdown(f"""
                <div class="prediction-card {prediction_class}">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span class="emoji-icon">{icon}</span>
                        <strong>{'Sarcastic' if row['predicted_sarcasm_tag'] == 'yes' else 'Not Sarcastic'}</strong>
                    </div>
                    <p style="margin: 0; font-style: italic;">"{row['body']}"</p>
                    <small>Author: {row.get('author', 'Unknown')}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Download option
            st.markdown("### 💾 Download Results")
            csv = input_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV with Predictions",
                data=csv,
                file_name='sarcasm_predictions.csv',
                mime='text/csv',
                help="Download the complete dataset with sarcasm predictions"
            )

elif "Input Text Query" in option:
    st.markdown("## ✍️ Single Text Analysis")
    st.markdown("Enter text to analyze for sarcasm detection.")
    
    # Text input with better styling
    user_input = st.text_area(
        "Enter your text here:",
        placeholder="Type or paste your text here...",
        height=150,
        help="Enter any text you want to analyze for sarcasm"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        predict_button = st.button("🔍 Analyze Text", use_container_width=True)
    
    if predict_button and user_input.strip():
        with st.spinner("Analyzing text..."):
            prediction = predict_sarcasm([user_input])[0]
            
            # Display result with enhanced styling
            prediction_class = "sarcastic" if prediction == 'yes' else "not-sarcastic"
            icon = "😏" if prediction == 'yes' else "😊"
            result_text = "Sarcastic" if prediction == 'yes' else "Not Sarcastic"
            
            st.markdown(f"""
            <div class="prediction-card {prediction_class}" style="text-align: center; padding: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
                <h2 style="margin: 0 0 1rem 0;">{result_text}</h2>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <p style="margin: 0; font-style: italic; font-size: 1.1rem;">"{user_input}"</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence indicators
            st.markdown("### 📊 Analysis Confidence")
            confidence_cols = st.columns(4)
            with confidence_cols[0]:
                st.metric("Text Length", len(user_input))
            with confidence_cols[1]:
                st.metric("Words", len(user_input.split()))
            with confidence_cols[2]:
                st.metric("Punctuation", sum(1 for c in user_input if c in '!?.,;:'))
            with confidence_cols[3]:
                st.metric("Capitalization", sum(1 for c in user_input if c.isupper()))

elif "View Evaluation Metrics" in option:
    st.markdown("## 📊 Model Evaluation Metrics")
    st.markdown("Comprehensive performance metrics for the sarcasm detection model.")
    
    # Overall accuracy card
    st.markdown("### 🎯 Overall Model Performance")

    # Metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Macro Avg F1", f"{macro_avg[2]:.2f}")
    with col3:
        st.metric("Weighted Avg F1", f"{weighted_avg[2]:.2f}")
    
    # Detailed classification report
    st.markdown("### 📋 Detailed Classification Report")
    
    # Create a styled dataframe
    report_df = pd.DataFrame({
        "Class": ["Not Sarcastic", "Sarcastic"],
        "Precision": training_report['precision'],
        "Recall": training_report['recall'],
        "F1-Score": training_report['f1-score'],
        "Support": training_report['support']
    })

    # Display with custom styling (requires matplotlib)
    st.dataframe(
        report_df.style.format({
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}'
        }).background_gradient(cmap='RdYlGn', subset=['Precision', 'Recall', 'F1-Score']),
        use_container_width=True
    )

    # Performance insights
    st.markdown("### 💡 Performance Insights")

    # Inject custom dark theme CSS for feature-box
    st.markdown("""
    <style>
    .feature-box {
        background-color: #1e1e1e;  /* dark box background */
        color: white;              /* white text */
        border-left: 5px solid #4CAF50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 2px 2px 8px rgba(255,255,255,0.05);
    }
    .feature-box h4 {
        margin-top: 0;
        color: #00e676; /* light green title */
    }
    .feature-box ul {
        margin: 0;
        padding-left: 20px;
    }
    .feature-box li {
        padding-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Insights section with dark themed boxes
    insight_cols = st.columns(2)
    with insight_cols[0]:
        st.markdown("""
        <div class="feature-box">
            <h4>✅ Strengths</h4>
            <ul>
                <li>High precision for non-sarcastic text (99%)</li>
                <li>Good recall for sarcastic text (86%)</li>
                <li>Robust feature extraction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with insight_cols[1]:
        st.markdown("""
        <div class="feature-box">
            <h4>⚠️ Areas for Improvement</h4>
            <ul>
                <li>Low precision for sarcastic text (8%)</li>
                <li>Class imbalance (83 vs 2611 samples)</li>
                <li>Need more sarcastic training data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Model architecture info
    st.markdown("### 🏗️ Model Architecture")
    st.markdown("""
    <div class="feature-box">
        <h4>🔧 Technical Details</h4>
        <ul>
            <li><strong>Algorithm:</strong> Machine Learning with TF-IDF + Custom Features</li>
            <li><strong>Features:</strong> Pragmatic, Explicit Incongruity, Implicit Incongruity</li>
            <li><strong>Vectorization:</strong> TF-IDF with custom preprocessing</li>
            <li><strong>Training Data:</strong> Reddit and Twitter datasets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

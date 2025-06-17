import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, sentiwordnet as swn, wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
import nltk
import re
from sklearn.metrics import classification_report

# Load the trained model and vectorizer
model = joblib.load('model_zero.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('wordnet')

# Initialize stop words and sentiment analyzer
stop_words = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

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

# Streamlit GUI
st.title("Nuanced Sentiment Detection")
st.sidebar.header("Options")

option = st.sidebar.selectbox("Choose input method:", ("Upload CSV File", "Input Text Query", "View Evaluation Metrics"))

if option == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Input Data")
        st.dataframe(input_data)
        
        # Prediction
        input_texts = input_data['body'].values
        predictions = predict_sarcasm(input_texts)
        input_data['predicted_sarcasm_tag'] = predictions
        
        # Display results with colors
        # st.write("Prediction Results")
        # def colorize(val):
        #     color = '#ffcccc' if val == 'yes' else '#ccffcc'
        #     return f'background-color: {color}'
        
        # results = input_data[['body', 'author', 'predicted_sarcasm_tag']]
        # styled_results = results.style.applymap(colorize, subset=['predicted_sarcasm_tag'])
        # st.write(styled_results)
        # Display results with colors
        st.write("Prediction Results")
        def colorize(row):
            color = '#ffcccc' if row['predicted_sarcasm_tag'] == 'yes' else '#ccffcc'
            return [f'background-color: {color}']*len(row)

        results = input_data[['body', 'author', 'predicted_sarcasm_tag']]
        styled_results = results.style.apply(colorize, axis=1)
        st.write(styled_results)

        
        # Download option
        st.write("Download Results")
        csv = input_data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name='predictions.csv', mime='text/csv')

elif option == "Input Text Query":
    user_input = st.text_area("Enter text for sarcasm detection:")
    if st.button("Predict"):
        prediction = predict_sarcasm([user_input])[0]
        st.write(f"Prediction: {'Sarcastic' if prediction == 'yes' else 'Not Sarcastic'}")
        st.markdown(f"<div style='background-color: {'#ffcccc' if prediction == 'yes' else '#ccffcc'}; padding: 10px;'><strong>{user_input}</strong></div>", unsafe_allow_html=True)

elif option == "View Evaluation Metrics":
    st.write("Training Classification Report")
    st.table(pd.DataFrame({
        "Class": ["no", "yes"],
        "Precision": training_report['precision'],
        "Recall": training_report['recall'],
        "F1-Score": training_report['f1-score'],
        "Support": training_report['support']
    }).set_index("Class"))
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Macro Avg**: Precision: {macro_avg[0]:.2f}, Recall: {macro_avg[1]:.2f}, F1-Score: {macro_avg[2]:.2f}")
    st.write(f"**Weighted Avg**: Precision: {weighted_avg[0]:.2f}, Recall: {weighted_avg[1]:.2f}, F1-Score: {weighted_avg[2]:.2f}")

# Display evaluation metrics in the sidebar
# st.sidebar.header("Model Evaluation Metrics")
# st.sidebar.text(f"Accuracy: {accuracy * 100:.2f}%")
# st.sidebar.text(f"Precision (No): {training_report['precision'][0] * 100:.2f}%")
# st.sidebar.text(f"Recall (No): {training_report['recall'][0] * 100:.2f}%")
# st.sidebar.text(f"F1 Score (No): {training_report['f1-score'][0] * 100:.2f}%")
# st.sidebar.text(f"Precision (Yes): {training_report['precision'][1] * 100:.2f}%")
# st.sidebar.text(f"Recall (Yes): {training_report['recall'][1] * 100:.2f}%")
# st.sidebar.text(f"F1 Score (Yes): {training_report['f1-score'][1] * 100:.2f}%")
# st.sidebar.text(f"Macro Avg - Precision: {macro_avg[0] * 100:.2f}%, Recall: {macro_avg[1] * 100:.2f}%, F1 Score: {macro_avg[2] * 100:.2f}%")
# st.sidebar.text(f

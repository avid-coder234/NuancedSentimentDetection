import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, sentiwordnet as swn, wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
import re
import nltk

# Load the test dataset
test_data = pd.read_csv('reddit_test.csv')

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('wordnet')

# Initialize stop words and sentiment analyzer
stop_words = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

# Load the trained model and vectorizer
model = joblib.load('model_zero.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def get_sentiment_scores(word):
    synsets = wn.synsets(word)
    if not synsets:
        return 0
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return swn_synset.pos_score() - swn_synset.neg_score()

def preprocess_test_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return ' '.join(tokens)

# Preprocess the test text data
test_data['clean_text'] = test_data['body'].apply(preprocess_test_text)

# Feature Extraction using TF-IDF
X_lexical_test = vectorizer.transform(test_data['clean_text']).toarray()

# Pragmatic features extraction
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

# Explicit Incongruity Features
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

# Implicit Incongruity Features
def extract_implicit_incongruity_features(text):
    verb_phrases = len([word for word in word_tokenize(text) if any(syn.pos() == 'v' for syn in wn.synsets(word))])
    noun_phrases = len([word for word in word_tokenize(text) if any(syn.pos() == 'n' for syn in wn.synsets(word))])
    return [verb_phrases, noun_phrases]

# Extract pragmatic, explicit, and implicit features for the test data
X_pragmatic_test = np.array([extract_pragmatic_features(text) for text in test_data['body']])
X_explicit_test = np.array([extract_explicit_incongruity_features(text) for text in test_data['body']])
X_implicit_test = np.array([extract_implicit_incongruity_features(text) for text in test_data['body']])

# Combine lexical and all other features
X_test = np.hstack((X_lexical_test, X_pragmatic_test, X_explicit_test, X_implicit_test))

# Predict with the loaded model
y_test_pred = model.predict(X_test)

# Add the predicted sarcasm tag to the test dataset
test_data['predicted_sarcasm_tag'] = y_test_pred

# Save predictions to a CSV file
test_data.to_csv('test_predictions.csv', index=False)

# Load the predictions file
predictions_data = pd.read_csv('test_predictions.csv')

# Count occurrences in the predictions file
occurrences_yes = (predictions_data['predicted_sarcasm_tag'] == 'yes').sum()
occurrences_no = (predictions_data['predicted_sarcasm_tag'] == 'no').sum()
print("Occurrences of 'yes' in the predicted_sarcasm_tag column:", occurrences_yes)
print("Occurrences of 'no' in the predicted_sarcasm_tag column:", occurrences_no)

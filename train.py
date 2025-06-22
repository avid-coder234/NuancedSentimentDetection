train.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.ensemble import EasyEnsembleClassifier
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, sentiwordnet as swn, wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
import re

# Load training data
train_data = pd.read_csv('reddit_training.csv')

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

def preprocess_and_extract_features(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
    clean_text = ' '.join(tokens)
    
    pragmatic_features = [
        any(char.isupper() for char in text),
        text.count('!'),
        text.count('?'),
        sum([1 for char in text if char in '.,;:']),
        sum([1 for word in tokens if word in [':)', ':(', ':-)', ':-(', ':D', 'XD', ':P', ':o', ':-D', ':-P', ':-o']]),
        text.lower().count('lol') + text.lower().count('haha') + text.lower().count('hehe'),
        sid.polarity_scores(text)['compound']
    ]
    
    pos_count = sum(get_sentiment_scores(word) > 0 for word in tokens)
    neg_count = sum(get_sentiment_scores(word) < 0 for word in tokens)
    sentiment_incongruities = abs(pos_count - neg_count)
    
    positive_words = [word for word in tokens if get_sentiment_scores(word) > 0]
    negative_words = [word for word in tokens if get_sentiment_scores(word) < 0]
    
    largest_pos_seq = max([len(seq) for seq in re.findall(r'(?:\b(?:' + '|'.join(positive_words) + r')\b\s*)+', text)], default=0)
    largest_neg_seq = max([len(seq) for seq in re.findall(r'(?:\b(?:' + '|'.join(negative_words) + r')\b\s*)+', text)], default=0)

    explicit_incongruity_features = [
        sentiment_incongruities,
        largest_pos_seq,
        largest_neg_seq,
        pos_count,
        neg_count
    ]
    
    verb_phrases = len([word for word in tokens if any(syn.pos() == 'v' for syn in wn.synsets(word))])
    noun_phrases = len([word for word in tokens if any(syn.pos() == 'n' for syn in wn.synsets(word))])
    implicit_incongruity_features = [verb_phrases, noun_phrases]

    features = np.concatenate((
        pragmatic_features,
        explicit_incongruity_features,
        implicit_incongruity_features
    ))
    return clean_text, features

# Preprocess and extract features for all training data
processed_texts = []
additional_features = []

for text in train_data['body']:
    clean_text, features = preprocess_and_extract_features(text)
    processed_texts.append(clean_text)
    additional_features.append(features)

# Fit the TfidfVectorizer on the entire corpus
lexical_vectorizer = TfidfVectorizer(max_features=5000)
X_lexical = lexical_vectorizer.fit_transform(processed_texts).toarray()

# Convert additional features to numpy array
additional_features = np.array(additional_features)

# Combine lexical and additional features
X = np.concatenate((X_lexical, additional_features), axis=1)
y = train_data['sarcasm_tag']

# Address class imbalance using EasyEnsemble
easy_ensemble = EasyEnsembleClassifier(random_state=42, n_estimators=10)
easy_ensemble.fit(X, y)
y_pred = easy_ensemble.predict(X)
print("Training Classification Report:\n", classification_report(y, y_pred))

# Save the model and vectorizer for later use
joblib.dump(easy_ensemble, 'model_zero.pkl')
joblib.dump(lexical_vectorizer, 'tfidf_vectorizer.pkl')


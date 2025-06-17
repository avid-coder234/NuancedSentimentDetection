# Nuanced Sentiment Detection using NLP Techniques

## Overview
This Python application, built with Streamlit, implements a sarcasm detection model based on the methodology outlined in a provided research paper. It preprocesses text data, trains a machine learning model, and classifies input text as sarcastic or non-sarcastic. The application leverages various natural language processing (NLP) techniques to enhance detection accuracy.

## Features

1. **User Interface**
   - Interactive web interface powered by Streamlit for intuitive user interaction.
   - Allows users to input text and receive real-time sarcasm detection results.

2. **Text Preprocessing**
   - **Tokenization**: Splits text into tokens using `word_tokenize` from the NLTK library.
   - **Lowercasing**: Converts text to lowercase to ensure uniformity.
   - **Punctuation and URL Removal**: Eliminates punctuation and URLs from the text.
   - **Stopword Removal**: Removes common stopwords using a combination of NLTK's stopwords and custom stopword lists.
   - **Stemming**: Reduces words to their root forms using Porter Stemmer.
   - **Word Splitting**: Splits long concatenated words using the `wordninja` library.

3. **Model Training**
   - **TF-IDF Vectorization**: Converts text data into numerical form using TF-IDF (Term Frequency-Inverse Document Frequency).
   - **Classification Algorithm**: Trains a machine learning model to classify text as sarcastic or non-sarcastic. The specific algorithm used is based on the research paper's methodology.
   - **Evaluation Metrics**: Provides accuracy, precision, recall, and F1-score metrics to evaluate the model's performance.
   - **Handling Imbalanced Dataset**: Uses the EasyEnsembleClassifier from imbalanced-learn to address the class imbalance between sarcastic and non-sarcastic samples.

4. **Real-time Sarcasm Detection**
   - Users can input text through the web interface to classify it as sarcastic or non-sarcastic.
   - Displays classification results and confidence scores.

## Libraries and Imports Used
- **Streamlit**: Framework for creating the web interface.
- **scikit-learn**: Machine learning library for TF-IDF vectorization and classification algorithms.
- **imblearn**: Provides the EasyEnsembleClassifier for handling class imbalance.
- **NLTK**: Natural Language Toolkit for text preprocessing tasks (tokenization, stopwords, stemming).
- **wordninja**: Library for splitting concatenated words.
- **os, re, numpy**: Standard Python libraries for file operations, regular expressions, and numerical operations.

## Applications
1. **Social Media Monitoring**: Detect sarcasm in social media posts to better understand user sentiments.
2. **Customer Service**: Identify sarcastic comments in customer feedback to improve service responses.
3. **Content Moderation**: Assist in moderating online content by detecting sarcastic remarks that might be offensive.

## How to Use

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   ```sh
   python -m nltk.downloader punkt stopwords
   ```

### Configuration
- Update directory paths and other configuration settings as needed in the `main` function according to your local setup.

### Running the App
1. Launch the app using Streamlit:
   ```sh
   streamlit run gui.py
   ```
2. Access the application through the provided local URL (typically `http://localhost:8501`).

### Usage

- **Choose Input Method:** Select between "Upload CSV File," "Input Text Query," or "View Evaluation Metrics" from the sidebar options.
  
- **Upload CSV File:**
  - Upload a CSV file containing text data for sarcasm detection.
  - View and interact with the uploaded data.
  - Obtain predictions for sarcasm tags and download the results as a CSV file.

- **Input Text Query:**
  - Enter text directly into the provided text area.
  - Click the "Predict" button to get an immediate sarcasm prediction.

- **View Evaluation Metrics:**
  - Access the training classification report.
  - View detailed metrics such as precision, recall, F1-score, support, and overall accuracy of the model.

### Handling Imbalanced Dataset

- **Class Imbalance**: The dataset used for training this model has a significant imbalance between sarcastic and non-sarcastic samples. To address this, the EasyEnsembleClassifier from the imbalanced-learn library was used.
  - **EasyEnsembleClassifier**: This method creates multiple balanced subsets of the training data by under-sampling the majority class and then trains a classifier on each subset. The final prediction is made by averaging the predictions of all classifiers.
  - **Effectiveness**: This approach helps in dealing with class imbalance, ensuring that the model is exposed to sufficient instances of the minority class (sarcastic samples) during training, improving the model's ability to correctly identify sarcasm.

### Conclusion

This sarcasm detection tool, built with Streamlit, offers a user-friendly interface for identifying sarcasm in text data. It combines advanced text processing techniques, sentiment analysis, and machine learning to provide accurate and insightful predictions. Whether uploading a dataset or inputting text directly, users can easily interact with the tool to detect sarcasm and review comprehensive evaluation metrics, making it useful for both educational and practical applications in natural language processing and sentiment analysis.

## License
This project is licensed under the MIT License.

## Contact
For inquiries or issues, please contact 

## Acknowledgements
- Streamlit: https://streamlit.io/
- scikit-learn: https://scikit-learn.org/
- imblearn: https://imbalanced-learn.org/
- NLTK: https://www.nltk.org/

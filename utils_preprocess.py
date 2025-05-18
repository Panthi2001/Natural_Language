# utils_preprocess.py
# Text cleaning and TF-IDF vectorization utilities for fake news classification.

import re
import string
import yaml
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load English stopwords once
try:
    EN_STOPWORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    EN_STOPWORDS = set(stopwords.words("english"))

# Initialize a lemmatizer
LEMMATIZER = WordNetLemmatizer()

def load_config(config_path="config.yaml"):
    """
    Load YAML configuration from the specified path.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def clean_text(text):
    """
    Perform text cleaning including:
      - Lowercasing
      - Removing HTML tags
      - Removing URLs
      - Removing punctuation and digits
      - Tokenization (keep only alphabetic characters)
      - Removing stopwords
      - Lemmatization

    Returns a cleaned string.
    """
    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', ' ', text)

    # Remove punctuation and digits
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize on whitespace
    tokens = text.split()

    # Remove stopwords and lemmatize
    cleaned_tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in EN_STOPWORDS and len(tok) > 1
    ]

    return ' '.join(cleaned_tokens)

def load_dataset(csv_path):
    """
    Load the FakeNewsNet dataset CSV into a pandas DataFrame.
    Assumes it has columns 'title', 'text', and 'label' (0 for real, 1 for fake).
    """
    df = pd.read_csv(csv_path)
    # If dataset has other naming conventions, adjust accordingly
    df = df.rename(columns={
        'title': 'title',
        'text': 'text',
        'label': 'label'  # Ensure labels are numeric 0 or 1
    })
    df = df.dropna(subset=['title', 'text', 'label'])
    df['combined'] = df['title'] + " " + df['text']
    return df[['combined', 'label']]

def preprocess_dataframe(df, text_column="combined"):
    """
    Clean all text entries in the DataFrame using clean_text().
    """
    df['cleaned'] = df[text_column].apply(clean_text)
    return df[['cleaned', 'label']]

def build_tfidf_vectorizer(config):
    """
    Build and return a TF-IDF vectorizer using configuration parameters.
    """
    tfidf_cfg = config['preprocessing']
    vectorizer = TfidfVectorizer(
        max_df=tfidf_cfg['max_df'],
        min_df=tfidf_cfg['min_df'],
        max_features=tfidf_cfg['max_vocab_size']
    )
    return vectorizer

def vectorize_texts(vectorizer, train_texts, test_texts):
    """
    Fit TF-IDF on train_texts and transform both train and test texts.
    Returns (X_train_tfidf, X_test_tfidf).
    """
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test

# If run as script, demonstrate preprocessing
if __name__ == "__main__":
    config = load_config()
    df = load_dataset(config['dataset']['fake_news_csv'])
    df_clean = preprocess_dataframe(df)
    print("Sample cleaned text:\n", df_clean.head())

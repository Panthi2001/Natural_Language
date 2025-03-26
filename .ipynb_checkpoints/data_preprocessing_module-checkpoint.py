# We define a preprocessing function that performs the following operations:

# Converting all text to lowercase

# Removing numbers from the text corpus

# Removing punctuation from the text corpus

# Removing special characters from the text corpus

# Removing english stopwords

# Stemming words to root words

# Removing extra whitespaces from the text corpus



import string                      # Provides common string operations (e.g., handling punctuation)
import nltk                        # Natural Language Toolkit for various NLP tasks
from nltk.tokenize import word_tokenize, WhitespaceTokenizer  # Functions to split text into words or by whitespace
from nltk.stem import PorterStemmer  # Tool to reduce words to their root form
from nltk.corpus import stopwords   # List of common words to filter out from text processing

nltk.download('stopwords')         # Download stopwords dataset if not already available
nltk.download('punkt_tab')         # Download tokenization models for splitting text (typically 'punkt' is used)
# Initialize tokenizer and stemmer
ps = PorterStemmer()                          # Initialize the Porter stemmer for reducing words to their base form
wst = WhitespaceTokenizer()                   # Initialize whitespace tokenizer to split text based on spaces

# 1. Convert text to lowercase
def lower_func(text):
    return text.lower()                       # Convert the entire text to lowercase

# # 2. Remove numbers from the text
# def remove_number_func(text): 
#     new_text = ''.join([char for char in text if not char.isdigit()])
#     return new_text                           # Remove all digit characters from the text

# 3. Remove punctuation
def remove_punc_func(text):
    new_text = ''.join([char for char in text if char not in string.punctuation])
    return new_text                           # Remove all punctuation characters using the string module

# 4. Remove special characters (keeping only alphanumeric and spaces)
def remove_spec_char_func(text):
    new_text = ''.join([char for char in text if char.isalnum() or char == ' '])
    return new_text                           # Keep only alphanumeric characters and spaces, removing others

# 5. Remove stopwords
def remove_stopwords(text):
    words = text.split()                      
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(filtered_words)           # Remove common English stopwords from the text

# 6. Apply stemming
def stem_func(text):
    words = word_tokenize(text)               
    stemmed_words = [ps.stem(word) for word in words]
    return " ".join(stemmed_words)             # Stem each word to its root form using the Porter stemmer

# 7. Remove extra whitespaces
def remove_whitespace_func(text):
    return " ".join(wst.tokenize(text))         # Tokenize to remove extra spaces and rejoin the text

# Function to apply all preprocessing steps sequentially
def preprocess_text(text):
    text = lower_func(text)                   # Step 1: Convert text to lowercase
    # text = remove_number_func(text)           # Step 2: Remove numbers
    text = remove_punc_func(text)             # Step 3: Remove punctuation
    text = remove_spec_char_func(text)        # Step 4: Remove special characters
    text = remove_stopwords(text)             # Step 5: Remove stopwords
    text = stem_func(text)                    # Step 6: Apply stemming
    text = remove_whitespace_func(text)       # Step 7: Remove extra whitespaces
    return text                               # Return the fully preprocessed text

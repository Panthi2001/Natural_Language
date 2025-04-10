
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

# Download resources if needed
nltk.download('punkt')
nltk.download('stopwords')
from heapq import nlargest

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def summarize(text):
    words = word_tokenize(text)

    #Word frequency
    word_frequency={}

    for word in words:
        lower_word = word.lower()
        if lower_word not in stop_words and word not in punctuations:
            word_count=word_frequency.get(word)
            if word_count is None:
                word_frequency[word]=1
            else:
                word_frequency[word]=word_count+1
            

    most_frequent = max(word_frequency, key=word_frequency.get)
    max_frequency=word_frequency[most_frequent]

    for word,frequency in word_frequency.items():
        word_frequency[word]= frequency/max_frequency

    sentences=sent_tokenize(text)

    scored_sentences={}

    for sentence in sentences:
        words = word_tokenize(sentence)
        sentence_score=0
        for word in words:
            word_score=word_frequency.get(word)
            if word_score is None:
                sentence_score+=0
            else:
                sentence_score+=word_score
        scored_sentences[sentence]=sentence_score


    summarized_length=1

    summary=nlargest(summarized_length,scored_sentences,key=scored_sentences.get) #highest scoring sentences 

    
    final_summary=" \n".join(summary)

    return final_summary







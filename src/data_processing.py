import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('sample')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Preprocesses text data by removing non-alphanumeric characters, converting to lowercase, removing stop words, and lemmatizing words.

def preprocess(text):

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    text = re.sub(r'\s+', ' ', text).strip()

    text = text.lower()

    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

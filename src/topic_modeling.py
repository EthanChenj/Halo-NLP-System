from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('sample')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# Initializes the TopicModeler with LDA. Ignores common words

class TopicModeler:
    def __init__(self, num_topics=5):  

        self.num_topics = num_topics
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=self.num_topics, random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.is_fitted = False

    def preprocess_text(self, text):

        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        words = [self.lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')]
        return ' '.join(words)

    # Trains LDA model. Transforms documents into matrix that represents term frequencies

    def fit(self, documents):

        preprocessed_documents = [self.preprocess_text(doc) for doc in documents]
        self.doc_term_matrix = self.vectorizer.fit_transform(preprocessed_documents)
        self.lda.fit(self.doc_term_matrix)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print("LDA model fitted.")
        self.is_fitted = True

    # Predicts topic distribution for new text

    def predict_topic(self, texts):

        if not self.is_fitted:
            raise Exception("LDA model not fitted. Call fit() first.")

        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        text_term_matrix = self.vectorizer.transform(preprocessed_texts)
        topic_probabilities = self.lda.transform(text_term_matrix)
        predicted_topics = [np.argmax(probs) for probs in topic_probabilities]
        return predicted_topics, topic_probabilities
    
    # Returns top keywords for topic

    def get_topic_details(self, topic_id):

        if not hasattr(self, 'feature_names'):
            raise Exception("Vectorizer not fitted.  Call fit() first.")
        if topic_id < 0 or topic_id >= self.num_topics:
            return "Invalid topic ID"

        topic_keywords = []
        topic_words = self.lda.components_[topic_id]
        top_word_indices = topic_words.argsort()[-10:][::-1]
        for i in top_word_indices:
            topic_keywords.append(self.feature_names[i])
        return topic_keywords
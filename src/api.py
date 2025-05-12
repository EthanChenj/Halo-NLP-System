from flask import Flask, request, jsonify, render_template
from sentiment_analysis import SentimentAnalyzer
from topic_modeling import TopicModeler
from data_processing import preprocess
import json

app = Flask(__name__)
sentiment_analyzer = SentimentAnalyzer()
topic_modeler = TopicModeler()

# Fit the topic model

with open('data/docs.json', 'r') as f: # Load from data/docs.json
        docs_data = json.load(f)
        docs = docs_data['docs']

topic_modeler.fit(docs)

# Main page

@app.route('/')
def index():
    return render_template('index.html')

# Analyzes the sentiment and topics of a review submitted

@app.route('/analyze', methods=['POST'])
def analyze_review():

    review_text = request.form.get('review_text')

    if not review_text:
        return jsonify({'error': 'Missing review_text in request'}), 400

    cleaned_review = preprocess(review_text)

    sentiment, sentiment_confidence = sentiment_analyzer.predict_sentiment(cleaned_review)
    topics, probs = topic_modeler.predict_topic([cleaned_review])
    topic_id = topics[0]
    topic_details = topic_modeler.get_topic_details(topic_id)

    response = {
        'review_text': review_text,
        'sentiment': sentiment,
        'sentiment_confidence': sentiment_confidence,
        'topic_id': int(topic_id),
        'topic_details': topic_details
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
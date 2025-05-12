from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

class SentimentAnalyzer:
    def __init__(self, model_name="bert-base-uncased", model_path='./models/sentiment_model'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            print(f"Loaded fine-tuned sentiment model from: {model_path}")
        except Exception as e:
            print(f"Failed to load fine-tuned model ({e}). Loading base model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.model.eval()
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

    # Takes text string as input and predicts its sentiment

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class_id].item()
            predicted_label = self.label_map.get(predicted_class_id)
        return predicted_label, confidence
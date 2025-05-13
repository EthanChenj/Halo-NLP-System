
# Halo Services: Sentiment Analysis and Topic Modeling

  

## Overview

  

Performs sentiment analysis and topic modeling on text data. This work was done as part of an internship at Halo Services. Predicts sentiment and identifies topics of future text (ex. reviews).

  

## Code Strucutre

  

- data: Contains the raw datasets.

- models: Contains the trained models.

- notebooks: Jupyter Notebooks used for exploration and training models.

- src: Contains scripts using models and API for analyzing reviews.

├── data # EXCLUDED FROM REPOSITORY

├── models # EXCLUDED FROM REPOSITORY

├── notebooks

│ ├── sentiment_analysis_training.ipynb

│ └── topic_modeling_exploration.ipynb

├── requirements.txt

├── src

│ └── api.py

│ ├── data_processing.py

│ ├── sentiment_analysis.py

│ └── topic_modeling.py

## Modeling Details

### Sentiment Analysis

- **Model**: The sentiment analysis uses the pre-trained transformer model 'bert-base-uncased', adjusted to the specific datasets to predict sentiment.
  
- **Fine-Tuning**: The sentiment_analysis_training.ipynb notebook uses the Hugging Face transformers library and PyTorch. The training includes:
  - **Tokenization**: The input text is tokenized with the pre-trained BERT model.
  - **Classification**: Adds a classification layer on top of the transformer's output to predict sentiment labels.
  - **Hyperparameter Tuning**: The training process is configured using TrainingArguments, which specifies hyperparameters like learning rate, batch size, and number of epochs.
  - **Evaluation**: Performance is evaluated using accuracy, precision, recall, and F1-score.

### Topic Modeling
- **Model**: The topic modeling component uses Latent Dirichlet Allocation (LDA) from scikit-learn. The TopicModeler class in topic_modeling.py is responsible for handling all the operations related to topic modeling, encapsulating this functionality.
- **LDA**: We use a probabilistic model for discovering underlying topics in a corpus in an unsupervised manner.
- **TF-IDF**: The input text documents are transformed into a document-term matrix before being fed into the LDA model, representing the importance of each word in each document.



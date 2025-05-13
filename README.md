# Halo Services: Sentiment Analysis and Topic Modeling

  

## Overview

  

This project implements a Natural Language Processing (NLP) pipeline for analyzing text data. Developed during an internship at Halo Services, this system uses machine learning techniques to predict the emotional tone (positive, negative, or neutral) and identify key discussion topics within unstructured text, such as customer reviews. The pipeline utilizes a transformer-based model for sentiment analysis and Latent Dirichlet Allocation (LDA) from scikit-learn for topic modeling. The output of this analysis provides valuable insights into customer opinions and trends, which can be used to improve services and decision-making.


  

## Code Strucutre

  

- ```data```: Contains the raw datasets.

- ```models```: Contains the trained models.

- ```notebooks```: Jupyter Notebooks used for exploration and training models.

- ```src```: Contains scripts using models and API for analyzing reviews.

```
├── data        # EXCLUDED FROM REPOSITORY
├── models      # EXCLUDED FROM REPOSITORY
├── notebooks
│   ├── sentiment_analysis_training.ipynb
│   └── topic_modeling_exploration.ipynb
├── requirements.txt
├── src
│   ├── api.py
│   ├── data_processing.py
│   ├── sentiment_analysis.py
│   └── topic_modeling.py
└── README.md
```

## Dependencies
```
Python 3.x
Transformers
PyTorch
Scikit-learn
NLTK
re
Pickle
```

## Modeling Details

### Sentiment Analysis:

- **Model**: The sentiment analysis uses the pre-trained transformer model ```bert-base-uncased```, adjusted to the specific datasets to predict sentiment.
  
- **Fine-Tuning**: The ```sentiment_analysis_training.ipynb``` notebook uses the Hugging Face transformers library and PyTorch. The training includes:
  - **Tokenization**: The input text is tokenized with the pre-trained BERT model.
  - **Classification**: Adds a classification layer on top of the transformer's output to predict sentiment labels.
  - **Hyperparameter Tuning**: The training process is configured using TrainingArguments, which specifies hyperparameters like learning rate, batch size, and number of epochs.
  - **Evaluation**: Performance is evaluated using accuracy, precision, recall, and F1-score.

### Topic Modeling:

- **Model**: The topic modeling component uses Latent Dirichlet Allocation (LDA) from scikit-learn. The ```TopicModeler``` class in ```topic_modeling.py``` is responsible for handling all the operations related to topic modeling, encapsulating this functionality.
- **Latent Dirichlet Allocation (LDA)**: We use a probabilistic model for discovering underlying topics in a corpus in an unsupervised manner.
- **Term Frequency-Inverse Document Frequency (TF-IDF)**: The input text documents are transformed into a document-term matrix before being fed into the LDA model, representing the importance of each word in each document.

## Interaction Between Notebooks and Scripts

This system combines Jupyter Notebooks and Python scripts to organize the workflow:

### Notebooks:

- Used for initial data exploration, experimentation, and model development.

### Scripts:

- Contains reusable code, organized into functions and classes.
- The ```src``` directory contains scripts that provides functions for text preprocessing, and defines the ```SentimentAnalyzer``` and ```TopicModeler``` classes for their respective analysis and modeling.
- Flask API is set up to expose the sentiment analysis and topic modeling functionalities.

### Workflow Summary:

- **Notebooks Develop Models**: The Notebooks are used to develop and train the machine learning models(sentiment analysis and topic modeling) and save the trained models to the models directory.
- **Scripts Use Models**: Python scripts then use these trained models for specific tasks. Loads the trained models from the models directory and uses them on a new set of text.
- **API Exposes Functionality**: The ```api.py``` script imports classes from the scripts and uses them to create an API, which can be used by other applications to analyze text.


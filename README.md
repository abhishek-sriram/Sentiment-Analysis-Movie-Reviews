# Sentiment Analysis - Movie Reviews

## Project Overview
This project aims to analyze movie reviews and classify them as **positive** or **negative** using Natural Language Processing (NLP) techniques. The dataset used for this project consists of movie reviews from IMDB, which are labeled with sentiment (either positive or negative). The model is trained using various machine learning classifiers, and the best performing model is saved for future predictions.

## Dataset
The dataset used in this project is the **IMDB Dataset** which consists of 50,000 movie reviews with their corresponding sentiments (positive/negative). You can download the dataset from [here](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Steps Involved
1. **Data Cleaning**:
    - Removal of HTML tags.
    - Removal of special characters.
    - Conversion to lowercase.
    - Removal of stopwords.
    - Stemming the words.

2. **Feature Extraction**:
    - Bag of Words (BOW) representation is used to convert text into a numerical format that the machine learning models can understand.

3. **Model Building**:
    - Three models are used:
        - Gaussian Naive Bayes
        - Multinomial Naive Bayes
        - Bernoulli Naive Bayes
    - The models are trained and evaluated based on accuracy to identify the best performing model.

4. **Model Evaluation**:
    - The accuracy of the models is compared, and the best performing model is saved using the `pickle` module.

## Libraries Used
```python
import numpy as np  # Linear algebra
import pandas as pd  # Data processing, CSV file I/O
import re  # Regex for text cleaning
from nltk.corpus import stopwords  # Stopwords for text processing
from nltk.tokenize import word_tokenize  # Tokenization of text
from nltk.stem import SnowballStemmer  # Stemming of words
from sklearn.feature_extraction.text import CountVectorizer  # Bag of Words model
from sklearn.model_selection import train_test_split  # Splitting data into training and testing sets
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB  # Naive Bayes classifiers
from sklearn.metrics import accuracy_score  # Model evaluation metric
import pickle  # For saving the model

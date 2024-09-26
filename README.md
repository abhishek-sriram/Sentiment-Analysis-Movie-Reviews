# **Sentiment Analysis - Movie Reviews**

## **Project Overview**
This project focuses on analyzing movie reviews and classifying them as either **positive** or **negative** using Natural Language Processing (NLP) techniques. By leveraging machine learning, the project seeks to provide an efficient method for sentiment analysis. The dataset comprises movie reviews from IMDB, labeled with corresponding sentiment scores. Multiple machine learning classifiers are trained, evaluated, and the best performing model is saved for future sentiment predictions.

## **Dataset**
The project utilizes the **IMDB Dataset**, containing 50,000 movie reviews, each labeled with either a positive or negative sentiment. You can download the dataset from [IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). 

## **Key Steps Involved**

1. **Data Cleaning**:
   - **HTML tag removal**: Cleansing reviews by eliminating unnecessary HTML content.
   - **Special character removal**: Stripping non-alphanumeric characters to simplify text.
   - **Lowercasing**: Ensuring uniformity by converting all text to lowercase.
   - **Stopword removal**: Filtering out common words (e.g., "the", "and") that provide little value for sentiment analysis.
   - **Stemming**: Reducing words to their root form to consolidate similar words.

2. **Feature Extraction**:
   - **Bag of Words (BOW)**: Transforming text data into a numerical matrix that represents word frequency, allowing machine learning algorithms to process and interpret the data.

3. **Model Building**:
   - The following Naive Bayes classifiers are employed:
     - **Gaussian Naive Bayes**
     - **Multinomial Naive Bayes**
     - **Bernoulli Naive Bayes**
   - Each model is trained on the dataset, and their performance is evaluated using accuracy scores to determine the most effective classifier.

4. **Model Evaluation & Saving**:
   - **Accuracy Comparison**: The models' performance is assessed based on accuracy scores.
   - **Model Saving**: The best performing model is saved using the `pickle` module for easy deployment and future predictions.

## **Technologies & Libraries Used**

The project is implemented using the following libraries:

```python
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis
import re  # Regular expressions for text cleaning
from nltk.corpus import stopwords  # To remove common stopwords
from nltk.tokenize import word_tokenize  # Tokenizing text into words
from nltk.stem import SnowballStemmer  # Stemming to reduce words to their root forms
from sklearn.feature_extraction.text import CountVectorizer  # BOW model for feature extraction
from sklearn.model_selection import train_test_split  # Splitting dataset into training and testing sets
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB  # Naive Bayes classifiers
from sklearn.metrics import accuracy_score  # For evaluating model performance
import pickle  # Saving the trained model for later use

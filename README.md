# **Sentiment Analysis - Movie Reviews**

## **Project Overview**
This project aims to analyze movie reviews and classify them as either **positive** or **negative** using Natural Language Processing (NLP) techniques. Leveraging machine learning algorithms, it offers an efficient method for sentiment analysis. The dataset consists of labeled movie reviews from IMDB, and multiple classifiers are trained to identify the best-performing model, which is then saved for future sentiment predictions.

## **Dataset**
The project utilizes the **IMDB Dataset**, comprising 50,000 movie reviews, each labeled with either a positive or negative sentiment. The dataset is available for download on [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## **Key Steps Involved**

1. **Data Cleaning**:
   - **HTML Tag Removal**: Cleanses the reviews by stripping out unnecessary HTML tags.
   - **Special Character Removal**: Removes non-alphanumeric characters for cleaner text representation.
   - **Lowercasing**: Converts all text to lowercase to maintain uniformity.
   - **Stopword Removal**: Filters out common, non-informative words (e.g., "the", "and") to focus on meaningful content.
   - **Stemming**: Reduces words to their root forms, consolidating different forms of the same word.

2. **Feature Extraction**:
   - **Bag of Words (BOW)**: Converts text data into a numerical matrix representing word frequencies, enabling machine learning algorithms to process the data.

3. **Model Building**:
     - Built an **Long Short Term Memory** model 
     - Models are trained on the dataset, and their performance is evaluated using accuracy scores.

4. **Model Evaluation & Saving**:
   - **Accuracy Comparison**: Models are compared based on their accuracy in classifying sentiments.
   - **Model Saving**: The best-performing model is serialized using the `pickle` module, allowing for easy reuse in future sentiment prediction tasks.

# Sentiment Analysis of Movie Reviews

## Overview
This project performs sentiment analysis on movie reviews from the IMDb dataset. The goal is to classify reviews as positive or negative using various machine learning algorithms. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and a simple command-line interface for sentiment prediction.

## Dataset
The dataset used in this project is the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle. It contains 50,000 movie reviews labeled as positive or negative.

## Project Steps
1. **Data Collection**: Download and load the IMDb movie reviews dataset.
2. **Data Preprocessing**:
   - Remove HTML tags and special characters.
   - Convert text to lowercase.
   - Lemmatize words and remove stopwords.
3. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of review sentiments.
   - Analyze the most frequent words in positive and negative reviews.
4. **Feature Engineering**:
   - Convert text data into numerical features using TF-IDF vectorization.
5. **Model Building**:
   - Train multiple machine learning models: Logistic Regression, Naive Bayes, Support Vector Machine, and Random Forest.
   - Evaluate models based on accuracy and classification reports.
6. **Model Evaluation**:
   - Compare the performance of different models.
   - Select the best-performing model.
7. **Prediction Interface**:
   - Implement a command-line interface for user input to predict sentiment.

## Usage
1. **Install Dependencies**:
   ```sh
   pip install pandas numpy nltk scikit-learn


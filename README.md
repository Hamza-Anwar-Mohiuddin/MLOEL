
# Sentiment Analysis Web Application

## Overview

This project is a web application for sentiment analysis of movie reviews using Logistic Regression and Naive Bayes models. The application processes and predicts the sentiment (Positive/Negative) of movie reviews using machine learning models. It includes manual implementations of Logistic Regression and Naive Bayes, as well as their counterparts from the `scikit-learn` library for comparison.

## Features

- **Manual Implementation of Logistic Regression**
- **Manual Implementation of Naive Bayes**
- **Integration with `scikit-learn`'s Logistic Regression and Naive Bayes**
- **Data Preprocessing including Lemmatization and Stop Words Removal**
- **TF-IDF Vectorization**
- **Model Training and Evaluation**
- **Flask Web Application for Sentiment Prediction**

## Prerequisites

- Python 3.7+
- Flask
- NLTK
- NumPy
- Pandas
- Scikit-learn
- Tqdm

## Usage

### Training Models

1. Ensure you have the `IMDBDataset.csv` file in the project directory.

2. Run the script to train the models and save the preprocessed data and models:



### Running the Web Application

1. Start the Flask application:

```bash
python app.py
```

2. Open your web browser and go to `http://127.0.0.1:5000`.

### Predicting Sentiment

Use the web interface to input movie reviews and get sentiment predictions. You can also use the `/predict` API endpoint with a POST request to predict sentiment programmatically.


## Project Structure

```bash
.
├── app.py                   # Flask web application
├── train.py                 # Script to train models
├── preprocess.py            # Data preprocessing functions
├── models.py                # Manual implementations of models
├── templates
│   └── index.html           # HTML template for the web application
├── SavedModels              # Directory to save/load models
│   ├── tfidf_vectorizer.pkl
│   ├── log_reg_model.pkl
│   └── nb_model.pkl
├── preprocessed_IMDBDataset.csv   # Preprocessed dataset
├── requirements.txt         # Required packages
└── README.md                # This file
```


## Acknowledgments

- The IMDB dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
- Special thanks to the developers of the libraries and tools used in this project.

---

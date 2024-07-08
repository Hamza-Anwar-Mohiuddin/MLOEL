import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os

# Integrate tqdm with pandas
tqdm.pandas()

# Uncomment the following lines if you need to download NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')

# Preprocess the data
print("Preprocessing data...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
    text = text.lower()  # Convert to lowercase
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Define functions for preprocessing input and predicting sentiment
def preprocess_input(review):
    review = preprocess_text(review)
    return vectorizer.transform([review]).toarray()

def predict_sentiment(review, model):
    review_vector = preprocess_input(review)
    prediction = model.predict(review_vector)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return sentiment

def main():
    global vectorizer  # Ensure vectorizer is accessible globally
    vectorizer = None
    logistic_reg = None
    naive_bayes = None
    logistic_reg_accuracy = 0
    nb_accuracy = 0

    if os.path.exists('../SavedModels/tfidf_vectorizer.pkl') and os.path.exists('../SavedModels/log_reg_model.pkl') and os.path.exists(
            '../SavedModels/nb_model.pkl'):
        print("Loading preprocessed data and models...")
        data = pd.read_csv('../preprocessed_IMDBDataset.csv')

        with open('../SavedModels/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            print("TF-IDF Vectorizer loaded successfully.")

        with open('../SavedModels/log_reg_model.pkl', 'rb') as f:
            logistic_reg = pickle.load(f)
            print("Logistic Regression model loaded successfully.")

        with open('../SavedModels/nb_model.pkl', 'rb') as f:
            naive_bayes = pickle.load(f)
            print("Naive Bayes model loaded successfully.")

        X = vectorizer.transform(data['review']).toarray()
        y = data['sentiment'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred_log_reg = logistic_reg.predict(X_test)
        logistic_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
        print(f"Logistic Regression:\n")
        print(f"Accuracy: {logistic_reg_accuracy}")
        print(classification_report(y_test, y_pred_log_reg))
        print("\n" + "="*60 + "\n")

        y_pred_nb = naive_bayes.predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_pred_nb)
        print(f"Naive Bayes:\n")
        print(f"Accuracy: {nb_accuracy}")
        print(classification_report(y_test, y_pred_nb))

    else:
        print("Loading raw dataset and training models...")
        data = pd.read_csv('../IMDBDataset.csv')
        data['review'] = data['review'].progress_apply(preprocess_text)
        data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

        print("Converting text data to numerical features using TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(data['review']).toarray()
        y = data['sentiment'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training Logistic Regression model using sklearn...")
        logistic_reg = LogisticRegression()
        logistic_reg.fit(X_train, y_train)
        y_pred_log_reg = logistic_reg.predict(X_test)
        logistic_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
        print(f"Logistic Regression:\n")
        print(f"Accuracy: {logistic_reg_accuracy}")
        print(classification_report(y_test, y_pred_log_reg))
        print("\n" + "="*60 + "\n")

        print("Training Naive Bayes model using sklearn...")
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train, y_train)
        y_pred_nb = naive_bayes.predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_pred_nb)
        print(f"Naive Bayes:\n")
        print(f"Accuracy: {nb_accuracy}")
        print(classification_report(y_test, y_pred_nb))

        # Save preprocessed data and models
        data.to_csv('preprocessed_IMDBDataset.csv', index=False)
        with open('../SavedModels/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open('../SavedModels/log_reg_model.pkl', 'wb') as f:
            pickle.dump(logistic_reg, f)
        with open('../SavedModels/nb_model.pkl', 'wb') as f:
            pickle.dump(naive_bayes, f)

    best_model = logistic_reg if logistic_reg_accuracy > nb_accuracy else naive_bayes

    while True:
        user_input = input("Enter a movie review (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        sentiment = predict_sentiment(user_input, best_model)
        print(f'Sentiment: {sentiment}')

if __name__ == "__main__":
    main()

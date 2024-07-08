import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

class ManualLogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000, regularization=0.1):
        self.lr = lr
        self.num_iter = num_iter
        self.regularization = regularization

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for i in tqdm(range(self.num_iter), desc="Training Logistic Regression"):
            linear_model = np.dot(X, self.theta)
            y_predicted = self._sigmoid(linear_model)
            gradient = np.dot(X.T, (y_predicted - y)) / y.size
            regularization_term = self.regularization * self.theta
            self.theta -= self.lr * (gradient + regularization_term)

    def predict(self, X):
        linear_model = np.dot(X, self.theta)
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class ManualNaiveBayes:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon  # Smoothing term to handle zero variance

    def fit(self, X, y):
        n_samples, n_features = X.shape
        print(f"X.shape {X.shape}")
        print(n_samples)
        print(n_features)
        self._classes = np.unique(y)
        print(f"self._classes {self._classes}")
        n_classes = len(self._classes)
        print(f"n_classes {n_classes}")

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        print(self._priors)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            print(X_c)
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0) + self.epsilon
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            print(self._mean)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

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
    manual_log_reg = None
    manual_nb = None
    logistic_reg_accuracy = 0
    nb_accuracy = 0

    if os.path.exists('../SavedModels/tfidf_vectorizer.pkl') and os.path.exists(
            '../SavedModels/manual_log_reg_model.pkl') and os.path.exists(
            '../SavedModels/manual_nb_model.pkl'):
        print("Loading preprocessed data and models...")
        data = pd.read_csv('../preprocessed_IMDBDataset.csv')

        with open('../SavedModels/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            print("TF-IDF Vectorizer loaded successfully.")

        with open('../SavedModels/manual_log_reg_model.pkl', 'rb') as f:
            manual_log_reg = pickle.load(f)
            print("Manual Logistic Regression model loaded successfully.")

        with open('../SavedModels/manual_nb_model.pkl', 'rb') as f:
            manual_nb = pickle.load(f)
            print("Manual Naive Bayes model loaded successfully.")

        X = vectorizer.transform(data['review']).toarray()
        y = data['sentiment'].values

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred_manual_log_reg = manual_log_reg.predict(X_test)
        logistic_reg_accuracy = accuracy_score(y_test, y_pred_manual_log_reg)
        print(f"Manual Logistic Regression:\n")
        print(f"Accuracy: {logistic_reg_accuracy}")
        print(classification_report(y_test, y_pred_manual_log_reg))
        print("\n" + "="*60 + "\n")

        y_pred_manual_nb = manual_nb.predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_pred_manual_nb)
        print(f"Manual Naive Bayes:\n")
        print(f"Accuracy: {nb_accuracy}")
        print(classification_report(y_test, y_pred_manual_nb))

    else:
        print("Loading raw dataset and training models...")
        data = pd.read_csv('../IMDBDataset.csv')
        data['review'] = data['review'].progress_apply(preprocess_text)
        data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

        print("Converting text data to numerical features using TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(data['review']).toarray()
        y = data['sentiment'].values

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training manual Logistic Regression model...")
        manual_log_reg = ManualLogisticRegression(lr=0.01, num_iter=1000, regularization=0.1)
        manual_log_reg.fit(X_train, y_train)
        y_pred_manual_log_reg = manual_log_reg.predict(X_test)
        logistic_reg_accuracy = accuracy_score(y_test, y_pred_manual_log_reg)
        print(f"Manual Logistic Regression:\n")
        print(f"Accuracy: {logistic_reg_accuracy}")
        print(classification_report(y_test, y_pred_manual_log_reg))
        print("\n" + "="*60 + "\n")

        print("Training manual Naive Bayes model...")
        manual_nb = ManualNaiveBayes()
        manual_nb.fit(X_train, y_train)
        y_pred_manual_nb = manual_nb.predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_pred_manual_nb)
        print(f"Manual Naive Bayes:\n")
        print(f"Accuracy: {nb_accuracy}")
        print(classification_report(y_test, y_pred_manual_nb))

        # Save preprocessed data and models
        data.to_csv('preprocessed_IMDBDataset.csv', index=False)
        with open('../SavedModels/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open('../SavedModels/manual_log_reg_model.pkl', 'wb') as f:
            pickle.dump(manual_log_reg, f)
        with open('../SavedModels/manual_nb_model.pkl', 'wb') as f:
            pickle.dump(manual_nb, f)

    best_model = manual_log_reg if logistic_reg_accuracy > nb_accuracy else manual_nb

    while True:
        user_input = input("Enter a movie review (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        sentiment = predict_sentiment(user_input, best_model)
        print(f'Sentiment: {sentiment}')

if __name__ == "__main__":
    main()


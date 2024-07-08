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
from flask import Flask, request, render_template, jsonify

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
    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

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
        epsilon = 1e-9  # Small value to avoid division by zero
        numerator = np.exp(- (x - mean) ** 2 / (2 * (var + epsilon)))
        denominator = np.sqrt(2 * np.pi * (var + epsilon))
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

def print_classification_report():
    global logistic_reg_accuracy, nb_accuracy
    print(f"Manual Logistic Regression:\n")
    print(f"Accuracy: {logistic_reg_accuracy}")
    print(classification_report(y_test, y_pred_manual_log_reg))
    print("\n" + "="*60 + "\n")
    print(f"Manual Naive Bayes:\n")
    print(f"Accuracy: {nb_accuracy}")
    print(classification_report(y_test, y_pred_manual_nb))

# Load models and data
def load_models():
    global vectorizer, manual_log_reg, manual_nb
    if os.path.exists('SavedModels/tfidf_vectorizer.pkl') and os.path.exists('SavedModels/manual_log_reg_model.pkl') and os.path.exists(
            'SavedModels/manual_nb_model.pkl'):
        with open('SavedModels/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('SavedModels/manual_log_reg_model.pkl', 'rb') as f:
            manual_log_reg = pickle.load(f)
        with open('SavedModels/manual_nb_model.pkl', 'rb') as f:
            manual_nb = pickle.load(f)
        return True
    return False

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review = data['review']
    sentiment = predict_sentiment(review, best_model)
    return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    report_printed = False  # Initialize the flag
    if load_models():
        data = pd.read_csv('preprocessed_IMDBDataset.csv')
        X = vectorizer.transform(data['review']).toarray()
        y = data['sentiment'].values

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred_manual_log_reg = manual_log_reg.predict(X_test)
        logistic_reg_accuracy = accuracy_score(y_test, y_pred_manual_log_reg)

        y_pred_manual_nb = manual_nb.predict(X_test)
        nb_accuracy = accuracy_score(y_test, y_pred_manual_nb)

        best_model = manual_log_reg if logistic_reg_accuracy > nb_accuracy else manual_nb

        # Print classification report and set the flag to True
        print_classification_report()
        report_printed = True

        app.run(debug=True)
    else:
        print("Models not found. Please run the training script to generate models.")

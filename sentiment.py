"""
Structure of the project

1) Importing necessary libraries: 
        Start by importing the required libraries, such as pandas, numpy, scikit-learn, 
        and NLTK (Natural Language Toolkit), which will be helpful for data manipulation, 
        feature extraction, and machine learning algorithms.

2) Data preprocessing: 
        Load your dataset and perform necessary preprocessing steps. 
        This typically includes removing any irrelevant information, handling missing values, 
        and cleaning the text by removing stopwords, punctuation, and converting it to lowercase. 
        You can use NLTK for these tasks.

3) Feature extraction: 
        Convert the preprocessed text into numerical features that can be used 
        for machine learning algorithms. Common techniques for feature extraction include Bag-of-Words (BoW), 
        TF-IDF (Term Frequency-Inverse Document Frequency), and word embeddings like Word2Vec or GloVe. 
        You can use libraries like scikit-learn or gensim for these tasks.

4) Splitting the data: 
        Split your dataset into training and testing sets. 
        The training set will be used to train your sentiment analysis model, 
        and the testing set will be used for evaluation.

5) Model selection and training: 
        Choose a suitable machine learning algorithm for sentiment analysis, 
        such as Naive Bayes, Support Vector Machines (SVM), or logistic regression. 
        Train your chosen model using the training data. Libraries like scikit-learn 
        provide implementations of these algorithms.

6) Model evaluation: 
        Evaluate the performance of your trained model on the testing set using 
        appropriate metrics like accuracy, precision, recall, or F1 score. 
        This will help you assess how well your model is performing.

7) Deploying the model: 
        Once you are satisfied with the model's performance, you can deploy it to make predictions on new data. 
        You can create a simple user interface or utilize APIs to accept input text 
        and generate sentiment predictions.

Additional tips:
- Keep in mind that there is no one-size-fits-all approach, and you can experiment with different techniques, algorithms, and parameters to improve your model's performance.
- Learning about cross-validation techniques, such as k-fold cross-validation, can help you obtain more reliable performance estimates for your model.
- Consider exploring advanced techniques like recurrent neural networks (RNNs) or transformers (such as BERT) for sentiment analysis, as they have shown promising results in recent years.

IMPORTANT: Give credit to the author of dataset (Chaithanya Kumar A Twitter and Reddit dataset)
"""
import numpy as np
import sys
import os
import csv
import time
from sklearn.model_selection import train_test_split

# Algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Naive bayes
from sklearn.naive_bayes import MultinomialNB
# SVC
from sklearn.svm import SVC
# Logistic regression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder



def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python sentiment.py data/file")
    
    try:
        dir, file = sys.argv[1].split("/")
    except ValueError:
        sys.exit("Usage: python sentiment.py data/file")

    # Load data section
    start_time_load = time.time()
    comments, labels = load_data(dir, file)
    end_time_load = time.time()
    print(f"\rLoad data > OK, runtime: {end_time_load - start_time_load}")

    # Feature extraction section
    start_time_bow = time.time()    
    bow = bag_of_words(comments)
    end_time_bow = time.time()
    print(f"\rFeature processing > OK, runtime: {end_time_bow - start_time_bow}")

    # Splitting data
    data = pd.read_csv('data/EcoPreprocessed.csv')
    label_encoder = LabelEncoder()
    data["division"] = label_encoder.fit_transform(data["division"])

    train_X, test_X, train_y, test_y = train_test_split(data["review"], data["division"], test_size=0.2, random_state=42)

    # Better algorithm LR=80% accuracy
    start_time_lr = time.time()    
    lr = logistic_regression(train_X, train_y, test_X, test_y)
    end_time_lr = time.time()
    print(f"\rAlgorithm processing > OK, runtime: {end_time_lr - start_time_lr}")
    print("\n Logistic regression:", lr)


def load_data(data, file_to_load):
    """
    Load the data from the directory data
    
    DATA ALREADY CLEANED
    DATA USED: EcoPrepocessed.csv (eco review on amazon)    
    """
    if os.path.exists(f"{data}/{file_to_load}"):
        print("Load data > Loading...", end="", flush=True)

        comments = ()
        label = ()
        with open(f"{data}/{file_to_load}") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                comments = comments + (row["review"],)
                label = label + (row["polarity"],)

            return (comments, label)
    else:
        sys.exit("No such file or directory, check the path or the file name")


def bag_of_words(comments):
    """
    Algorithm used for feature extraction of comments
    Algorithm used: BoW (Bag of Words)
    """
    print("Feature processing > Loading...", end="", flush=True)

    vocabulary = set()
    for comment in comments:
        words = comment.split(" ")
        vocabulary.update(words)
    
    vocabulary = sorted(vocabulary)

    feature_matrix = []
    for document in comments:
        words = document.split(" ")
        document_vector = [0] * len(vocabulary)
        
        for word in words:
            if word in vocabulary:
                word_index = vocabulary.index(word)
                document_vector[word_index] += 1

        feature_matrix.append(document_vector)
    
    feature_matrix = np.array(feature_matrix)
    return feature_matrix


def naive_bayes(train_X, train_y, test_X, test_y):
    print("Naive bayes > Loading...", end="", flush=True)
    classifier = MultinomialNB()
    classifier.fit(train_X, train_y)

    # Make prediction on the test set
    predictions = classifier.predict(test_X)

    # Calculate accuracy of the classifier
    accuracy = accuracy_score(test_y, predictions)
    return "Accuracy:", accuracy


def SVM(train_X, train_y, test_X, test_y):
    print("SVM > Loading...", end="", flush=True)
    classifier = SVC(kernel='sigmoid')
    classifier.fit(train_X, train_y)

    # Make predictions
    predictions = classifier.predict(test_X)

    # Calculate accuracy of the classifier
    accuracy = accuracy_score(test_y, predictions)
    return "Accuracy:", accuracy


def logistic_regression(train_X, train_y, test_X, test_y):
    vectorizer = TfidfVectorizer()
    train_x_vec = vectorizer.fit_transform(train_X)
    test_x_vec = vectorizer.transform(test_X)

    model = LogisticRegression()
    model.fit(train_x_vec, train_y)

    y_pred = model.predict(test_x_vec)

    accuracy = accuracy_score(test_y, y_pred)
    return "Accuracy:", accuracy

if __name__ == "__main__":
    main() 
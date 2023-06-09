"""
IMPORTANT: Give credit to the author of dataset (Chaithanya Kumar A Twitter and Reddit dataset)
"""
import numpy as np
import sys
import os
import time
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
import multiprocessing
from string import punctuation

# Logistic regression
from sklearn.linear_model import LogisticRegression

# Sequential neural network
import tensorflow as tf
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping


PS = nltk.stem.PorterStemmer()


class Algorithms:
    def __init__(self, comments, labels, vocabulary):
        self.comments = comments
        self.labels = labels
        self.vocabulary = vocabulary

        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(labels)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.comments, self.labels, test_size=0.2, random_state=42)
        
        self.train_x_vec = bag_of_words(self.train_X, self.vocabulary)
        self.test_x_vec = bag_of_words(self.test_X, self.vocabulary)

    def logistic_regression(self):
        model = LogisticRegression(solver="sag", max_iter=10000)
        model.fit(self.train_x_vec, self.train_y)

        y_pred = model.predict(self.test_x_vec)

        accuracy = f"\n  f1_score: {f1_score(self.test_y, y_pred)}\n  accuracy_score: {accuracy_score(self.test_y, y_pred)}\n  recall_score: {recall_score(self.test_y, y_pred)}"

        return (accuracy, "Logistic Regression")

    def Sequential_NN(self):
        # Tokenize the text and convert it into sequences
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(self.train_X)
        X_train_seq = tokenizer.texts_to_sequences(self.train_X)
        X_test_seq = tokenizer.texts_to_sequences(self.test_X)

        # Pad sequences to ensure equal length
        X_train_pad = pad_sequences(X_train_seq, maxlen=100)
        X_test_pad = pad_sequences(X_test_seq, maxlen=100)

        callback = EarlyStopping(monitor='loss', patience=2)

        # Create neural network model
        model = Sequential()
        model.add(Embedding(10000, 100, input_length=100))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.3)),
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_pad, self.train_y, epochs=3, batch_size=40, callbacks=callback)

        loss, accuracy = model.evaluate(X_test_pad, self.test_y)

        model.save('sentiment_model.h5')

        return (accuracy, "Sequential Neural Network")


class UserInterface:
    def __init__(self, user_input, vocabulary=None, comments=None, labels=None):
        self.vocabulary = vocabulary
        self.comments = comments
        self.labels = labels
        self.user_input = user_input
    
    def preprocess_text(self, text):
        stopwords = nltk.corpus.stopwords.words("english")
        words = nltk.tokenize.word_tokenize(text)

        filtered_words = []
        for word in words:
            if word not in stopwords:
                if word not in punctuation:
                    word = PS.stem(word)

                    filtered_words.append(word)
        
        preprocessed_text = ' '.join(filtered_words)
        return preprocessed_text

    def prediction_logistic_regression(self):
        preprocessed_input = bag_of_words([self.user_input], self.vocabulary)

        model = LogisticRegression(solver='sag', max_iter=10000)

        model.fit(bag_of_words(self.comments, self.vocabulary), self.labels)
        prediction = model.predict(preprocessed_input)

        return prediction[0]

    tf.function(reduce_retracing=True)
    def prediction_neural_network(self):
        model = load_model('sentiment_model.h5', compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        preprocessed_input = self.preprocess_text(self.user_input)

        # Convert preprocessed input into sequences
        texts = [preprocessed_input]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        input_seq = tokenizer.texts_to_sequences(texts)

        input_pad = pad_sequences(input_seq, maxlen=100)

        prediction = model(input_pad)

        # Interpret the prediction
        sentiment = "Positive" if prediction[0] >= 0.5 else "Negative"

        return sentiment


def main():
    dir, file = check_errors()

    try:
        algorithm_to_use = int(input("1) Logistic regression\n2) Sequential Neural Network\nWhich algorithm do you want to use?(1-2) "))
    except ValueError:
        sys.exit("\nError: You must choose one of three options with the corresponding number")

    if algorithm_to_use not in [1, 2]:
        sys.exit("\nError: No options match the number you entered")
    elif algorithm_to_use == 2 and os.path.exists("sentiment_model.h5"):
        while True:
            try:
                user_input = input("\n\nSentence: ").lower()
            except KeyboardInterrupt:
                sys.exit("\nSession terminated")

            user_interface = UserInterface(user_input)
            if algorithm_to_use == 2:
                print(f"> Sentiment {user_interface.prediction_neural_network()}")


    # Load data section
    print("\n\rLoad data...", end="", flush=True)
    comments, labels = load_data(dir, file)
    print(f"\rLoad data...OK")

    # Create a vocabulary
    print("\rVocabulary creation...", end="", flush=True)
    vocabulary = creation_vocabulary(comments)
    print(f"\rVocabulary creation...OK")
    
    # Algorithm sections
    print("\rAlgorithm process...", end="", flush=True)
    algorithms = Algorithms(comments, labels, vocabulary)
    if algorithm_to_use == 1:
        algorithm_accuracy, label = algorithms.logistic_regression()
    else:
        algorithm_accuracy, label = algorithms.Sequential_NN()
    print(f"\rAlgorithm process...OK")

    print(f"\n {label}: {algorithm_accuracy}")
    
    while True:
        try:
            user_input = input("\n\nSentence: ").lower()
        except KeyboardInterrupt:
            sys.exit("\nSession terminated")

        user_interface = UserInterface(user_input, vocabulary, comments, labels)
        if algorithm_to_use == 2:
            print(f"> Sentiment {user_interface.prediction_neural_network()}")
        elif algorithm_to_use == 1:
            print(f"> Sentiment {user_interface.prediction_logistic_regression()}")


def check_errors():
    if len(sys.argv) != 2:
        sys.exit("Usage: python sentiment.py data/file")
    
    try:
        dir, file = sys.argv[1].split("/")
    except ValueError:
        sys.exit("Usage: python sentiment.py data/file")

    if not os.path.exists(f"{dir}/{file}"):
        sys.exit("No such file or directory, check the path or the file name")

    return (dir, file)


def load_data(data, file_to_load):
    """
    Load the data from the directory data    
    """
    with open(f'{data}/{file_to_load}', 'r', encoding='utf-8', errors='ignore') as file:
        dataset = pd.read_csv(file)

    comments = tuple(dataset["Review"].values.tolist())
    labels = tuple(dataset["Freshness"].values.tolist())

    return (comments, labels)


def creation_vocabulary(comments):
    vocabulary = set()
    stopwords = nltk.corpus.stopwords.words("english")

    for comment in comments:
        words = nltk.word_tokenize(comment.lower())
        for word in words:
            if word not in stopwords:

                if word not in punctuation:
                    word = PS.stem(word)

                    vocabulary.update(word)
        
    return sorted(vocabulary)


def extract_features(document, vocabulary):
    words = nltk.word_tokenize(document.lower())
    document_vector = np.zeros(len(vocabulary))

    for word in words:
        word = PS.stem(word)
        if word in vocabulary:
            word_index = vocabulary.index(word)
            document_vector[word_index] += 1

    return document_vector


def bag_of_words(comments, vocabulary):
    """
    Algorithm used for feature extraction of comments: BoW

    Using multiprocessing for faster process
    """
    pool = multiprocessing.Pool()

    # Use the pool to parallelize the feature extraction
    feature_matrix = pool.starmap(extract_features, [(document, vocabulary) for document in comments])

    # Close the pool to release resources
    pool.close()
    pool.join()

    feature_matrix = [feature_matrix]
    return feature_matrix


if __name__ == "__main__":
    main() 
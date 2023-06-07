"""
IMPORTANT: Give credit to the author of dataset (Chaithanya Kumar A Twitter and Reddit dataset)
"""
import numpy as np
import sys
import os
import csv
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import nltk
import multiprocessing
from string import punctuation

# Logistic regression
from sklearn.linear_model import LogisticRegression

# Sequential neural network
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

# Installation of module from nltk
nltk.download('stopwords')
nltk.download('punkt')


PS = nltk.stem.PorterStemmer()


class Algorithms:
    def __init__(self, comments, labels, vocabulary):
        print("\rAlgorithm process > Loading...")

        self.comments = comments
        self.labels = labels
        self.vocabulary = vocabulary

        # Split data
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(labels)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.comments, self.labels, test_size=0.4, random_state=42)
        
        self.train_x_vec = bag_of_words(self.train_X, self.vocabulary)
        self.test_x_vec = bag_of_words(self.test_X, self.vocabulary)

    def logistic_regression(self):
        model = LogisticRegression(solver='newton-cholesky', max_iter=10000)
        model.fit(self.train_x_vec, self.train_y)

        y_pred = model.predict(self.test_x_vec)

        accuracy = accuracy_score(self.test_y, y_pred)

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

        callback = EarlyStopping(monitor='loss', patience=3)

        # Create a simple neural network model
        model = Sequential()
        model.add(Embedding(10000, 100, input_length=100))
        model.add(Flatten())
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.4)),
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_pad, self.train_y, epochs=10, batch_size=40, callbacks=callback)

        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test_pad, self.test_y)

        # Save the trained model
        model.save('sentiment_model.h5')

        return (accuracy, "Sequential Neural Network")


class UserInterface:
    def __init__(self, vocabulary=None, comments=None, labels=None):
        self.vocabulary = vocabulary
        self.comments = comments
        self.labels = labels

    def interface(self, algorithm_to_use):
        try:
            self.user_input = input("\n\nSentence: ").lower()
        except KeyboardInterrupt:
            sys.exit("\nSession terminated")

        if algorithm_to_use == 2:
            print(f"> Sentiment {self.prediction_neural_network()}")
        elif algorithm_to_use == 1:
            print(f"> Sentiment {self.prediction_logistic_regression()}")
    
    def preprocess_text(self, text):
        stopwords = nltk.corpus.stopwords.words("english")
        words = nltk.tokenize.word_tokenize(text)

        filtered_words = []
        for word in words:
            if word not in stopwords:
                filtered_words.append(word)

        stemmed_words = [PS.stem(word) for word in filtered_words]
        no_punctuation_words = [word for word in stemmed_words if word not in punctuation]
        
        preprocessed_text = ' '.join(no_punctuation_words)
        return preprocessed_text

    def prediction_logistic_regression(self):
        preprocessed_input = bag_of_words([self.user_input], self.vocabulary)

        model = LogisticRegression(solver='newton-cholesky', max_iter=10000)

        model.fit(bag_of_words(self.comments, self.vocabulary), self.labels)
        prediction = model.predict(preprocessed_input)

        return prediction[0]

    def prediction_neural_network(self):
        # Load the trained model
        model = load_model('sentiment_model.h5')

        # Preprocess the user input
        preprocessed_input = self.preprocess_text(self.user_input)

        # Convert preprocessed input into sequences
        texts = [preprocessed_input]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        input_seq = tokenizer.texts_to_sequences(texts)

        # Pad the sequence
        input_pad = pad_sequences(input_seq, maxlen=100)

        # Make prediction
        prediction = model.predict(input_pad)

        # TODO: Check for neutral of negative or positive

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
            user_interface = UserInterface()
            user_interface.interface(algorithm_to_use)


    # Load data section
    start_time_load = time.time()
    comments, labels = load_data(dir, file)
    end_time_load = time.time()
    print()
    print(f"\rLoad data > OK, runtime: {end_time_load - start_time_load}")

    # Create a vocabulary
    print("\rVocabulary creation > Loading...", end="", flush=True)
    start_time_vocabulary = time.time()

    vocabulary = creation_vocabulary(comments)

    end_time_vocabulary = time.time()
    print(f"\rVocabulary creation > OK, runtime: {end_time_vocabulary - start_time_vocabulary}")
    
    # Algorithm sections
    start_time_alg = time.time()
    
    algorithms = Algorithms(comments, labels, vocabulary)
    if algorithm_to_use == 1:
        algorithm_accuracy, label = algorithms.logistic_regression()
    else:
        algorithm_accuracy, label = algorithms.Sequential_NN()

    end_time_alg = time.time()
    # Move on line up the cursor
    sys.stdout.write("\033[F")
    print(f"\rAlgorithm process > OK, runtime: {end_time_alg - start_time_alg}")
    # Clean the line
    sys.stdout.write("\033[K")
    print(f"\n {label}: {algorithm_accuracy}")
    
    while True:
        user_interface = UserInterface(vocabulary, comments, labels)
        user_interface.interface(algorithm_to_use)


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
    comments = ()
    label = ()

    with open(f"{data}/{file_to_load}") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            comments = comments + (row["review"],)
            label = label + (row["sentiment"],)

        return (comments, label)


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
    # Create a multiprocessing pool
    pool = multiprocessing.Pool()

    # Use the pool to parallelize the feature extraction
    feature_matrix = pool.starmap(extract_features, [(document, vocabulary) for document in comments])

    # Close the pool to release resources
    pool.close()
    pool.join()

    feature_matrix = np.array(feature_matrix)
    return feature_matrix


if __name__ == "__main__":
    main() 
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

# Algorithms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import nltk
import multiprocessing
from string import punctuation
import sklearn

# Naive bayes
from sklearn.naive_bayes import MultinomialNB
# SVC
from sklearn.svm import SVC
# Logistic regression
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.preprocessing.text import Tokenizer

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

        # self.train_x_vec = sklearn.preprocessing.scale(self.train_x_vec)
        # self.test_x_vec = sklearn.preprocessing.scale(self.test_x_vec)

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

        # Create a simple neural network model
        model = Sequential()
        model.add(Embedding(10000, 100, input_length=100))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25)),
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_pad, self.train_y, epochs=10, batch_size=32)

        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test_pad, self.test_y)

        # Save the trained model
        model.save('sentiment_model.h5')

        return (accuracy, "Sequential Neural Network")


class UserInterface:
    def __init__(self, user_input, vocabulary, comments, labels, algorithm_to_use):
        self.user_input = user_input 
        self.vocabulary = vocabulary
        self.comments = comments
        self.labels = labels
        self.algorithm_to_use = algorithm_to_use

    def prediction(self):
        preprocessed_input = bag_of_words([self.user_input], self.vocabulary)

        model = LogisticRegression(solver='newton-cholesky', max_iter=10000)

        model.fit(bag_of_words(self.comments, self.vocabulary), self.labels)
        prediction = model.predict(preprocessed_input)

        return prediction[0]

    def prediction_neural_network(self):
        # Load the trained model
        model = load_model('sentiment_model.h5')

        # Preprocess the user input
        preprocessed_input = bag_of_words([self.user_input], self.vocabulary)  # Replace 'preprocess' with your own preprocessing function

        # Convert preprocessed input into sequences
        input_seq = Tokenizer.texts_to_sequences(preprocessed_input)

        # Pad the sequence
        input_pad = pad_sequences(input_seq, maxlen=100)

        # Make prediction
        prediction = model.predict(input_pad)

        # Interpret the prediction
        sentiment = "Positive" if prediction[0] >= 0.5 else "Negative"

        return sentiment


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python sentiment.py data/file")
    
    try:
        dir, file = sys.argv[1].split("/")
    except ValueError:
        sys.exit("Usage: python sentiment.py data/file")

    if not os.path.exists(f"{dir}/{file}"):
        sys.exit("No such file or directory, check the path or the file name")
    
    try:
        algorithm_to_use = int(input("\n1) Logistic regression\n2)Sequential Neural Network\nWhich algorithm do you want to use?(1-4) "))
    except ValueError:
        sys.exit("\nError: You must choose one of three options with the corresponding number")

    if algorithm_to_use not in [1, 2]:
        sys.exit("\nError: No options match the number you entered")


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

    # User input section
    while True:
        try:
            user_input = input("\n\nSentence: ").lower()
        except KeyboardInterrupt:
            sys.exit("\nSession terminated")

        user_interface = UserInterface(user_input, vocabulary, comments, labels, algorithm_to_use)
        if algorithm_to_use == 2:
            predicted = user_interface.prediction_neural_network()
        else:
            predicted = user_interface.prediction()

        print(f"> Sentiment {predicted}")


def load_data(data, file_to_load):
    """
    Load the data from the directory data
    
    DATA ALREADY CLEANED
    DATA USED: EcoPrepocessed.csv (eco review on amazon)    
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

                for letter in word:
                    if letter in punctuation:
                        word = word.replace(letter, "")

                vocabulary.update(word)
    return sorted(vocabulary)


def extract_features(document, vocabulary):
    words = nltk.word_tokenize(document.lower())
    document_vector = np.zeros(len(vocabulary))

    for word in words:
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
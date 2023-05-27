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

# Naive bayes
from sklearn.naive_bayes import MultinomialNB
# SVC
from sklearn.svm import SVC
# Logistic regression
from sklearn.linear_model import LogisticRegression


class Algorithms:
    def __init__(self, comments, labels, vocabulary):
        print("\rAlgorithm process > Loading...")

        self.comments = comments
        self.labels = labels
        self.vocabulary = vocabulary

        # Split data
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(labels)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.comments, self.labels, test_size=0.2, random_state=42)
        
        self.train_x_vec = bag_of_words(self.train_X, self.vocabulary)
        self.test_x_vec = bag_of_words(self.test_X, self.vocabulary)

    def naive_bayes(self):        
        classifier = MultinomialNB()
        classifier.fit(self.train_x_vec, self.train_y)

        # Make prediction on the test set
        predictions = classifier.predict(self.test_x_vec)

        # Calculate accuracy of the classifier
        accuracy = accuracy_score(self.test_y, predictions)
        return (accuracy, "Naive Bayes")

    def SVM(self):        
        classifier = SVC(kernel='sigmoid')
        classifier.fit(self.train_x_vec, self.train_y)

        predictions = classifier.predict(self.test_x_vec)

        accuracy = accuracy_score(self.test_y, predictions)
        return (accuracy, "Support Vectors Machine")

    def logistic_regression(self):
        model = LogisticRegression(solver='liblinear')
        model.fit(self.train_x_vec, self.train_y)

        y_pred = model.predict(self.test_x_vec)

        accuracy = accuracy_score(self.test_y, y_pred)
        return (accuracy, "Logistic Regression")


class UserInterface:
    def __init__(self, user_input, vocabulary, comments, labels, algorithm_to_use):
        self.user_input = user_input 
        self.vocabulary = vocabulary
        self.comments = comments
        self.labels = labels
        self.algorithm_to_use = algorithm_to_use

    def prediction(self):
        preprocessed_input = bag_of_words([self.user_input], self.vocabulary)

        # Predict sentiment
        if self.algorithm_to_use == 1:
            model = MultinomialNB()
        elif self.algorithm_to_use == 2:
            model = SVC(kernel='sigmoid')
        else:
            model = LogisticRegression(solver='liblinear')

        model.fit(bag_of_words(self.comments, self.vocabulary), self.labels)
        prediction = model.predict(preprocessed_input)

        return prediction[0]

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python sentiment.py data/file")
    
    try:
        dir, file = sys.argv[1].split("/")
    except ValueError:
        sys.exit("Usage: python sentiment.py data/file")

    algorithm_to_use = int(input("1) Naive bayes\n2) Support Vectore Machine\n3) Logistic regression\nWhich algorithm do you want to use?(1-3) "))

    # Load data section
    start_time_load = time.time()
    comments, labels = load_data(dir, file)
    end_time_load = time.time()
    print(f"\rLoad data > OK, runtime: {end_time_load - start_time_load}")

    # Create a vocabulary
    vocabulary = set()
    for comment in comments:
        words = comment.split(" ")
        vocabulary.update(words)
    vocabulary = sorted(vocabulary)
    
    # Algorithm sections
    start_time_alg = time.time()
    
    algorithms = Algorithms(comments, labels, vocabulary)
    if algorithm_to_use == 1:
        algorithm_accuracy, label = algorithms.naive_bayes()
    elif algorithm_to_use == 2:
        algorithm_accuracy, label = algorithms.SVM()
    else:
        algorithm_accuracy, label = algorithms.logistic_regression()

    end_time_alg = time.time()
    # Move on line up the cursor
    sys.stdout.write("\033[F")
    print(f"\rAlgorithm process > OK, runtime: {end_time_alg - start_time_alg}")
    # Clean the line
    sys.stdout.write("\033[K")
    print(f"\n {label}: {algorithm_accuracy}")

    # User input section
    print()
    while True:
        try:
            user_input = input("\nSentence: ")
        except KeyboardInterrupt:
            sys.exit("\nSession terminated")

        user_interface = UserInterface(user_input, vocabulary, comments, labels, algorithm_to_use)
        predicted = user_interface.prediction()
        print(f"> Sentiment {predicted}")

def load_data(data, file_to_load):
    """
    Load the data from the directory data
    
    DATA ALREADY CLEANED
    DATA USED: EcoPrepocessed.csv (eco review on amazon)    
    """
    if os.path.exists(f"{data}/{file_to_load}"):
        print("\nLoad data > Loading...", end="", flush=True)

        comments = ()
        label = ()
        with open(f"{data}/{file_to_load}") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                comments = comments + (row["review"],)
                label = label + (row["division"],)

            return (comments, label)
    else:
        sys.exit("No such file or directory, check the path or the file name")


def bag_of_words(comments, vocabulary):
    """
    Algorithm used for feature extraction of comments
    Algorithm used: BoW (Bag of Words)
    """
    vocabulary = vocabulary

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


if __name__ == "__main__":
    main() 
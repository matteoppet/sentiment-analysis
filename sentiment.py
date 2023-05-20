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
import numpy
import nltk
import sys
import os
import csv


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python sentiment.py data/file")
    
    try:
        dir, file = sys.argv[1].split("/")
    except ValueError:
        sys.exit("Usage: python sentiment.py data/file")

    comments, label = load_data(dir, file)
    print("\rLoad data completed")


def load_data(data, file_to_load):
    """
    Load the data from the directory data
    
    DATA ALREADY CLEANED
    """
    if os.path.exists(f"{data}/{file_to_load}"):
        print("Load data...", end="", flush=True)

        if file_to_load == "Reddit_Data.csv": 
            first_category = "clean_comment"
        else: 
            first_category = "clean_text"
       
        comments = ()
        label = ()
        with open(f"{data}/{file_to_load}") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                comments = comments + (row[first_category],)
                label = label + (row["category"],)


            return (comments, label)
    else:
        sys.exit("No such file or directory, check the path or the file name")

main() 
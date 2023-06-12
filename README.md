# Sentiment analysis
This sentiment analysis program is an AI that can recognize if your comment is positive, negative or neutral.
This program is also implemented with two different algorithms that you can choose when running it:
Logistic Regression and Sequential Neural Network. 
The neural network model is already saved, it will be updated in case of changes but in any case, if the program does not find the model, the neural network will be trained and it will take a few minutes before you can enter your comment, if it already exists you will be redirected directly to the request for comment.

> The "sentiment_model.h5" model file must be at the same level as the "sentiment.py" file

## Demo
![Demo photo](https://github.com/matteoppet/sentiment-analysis/photo/demo.png?raw=true)
Algorithm used in this image: Sequential Neural Network(https://keras.io/guides/sequential_model/)

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the libraries used.

```bash
pip install -r requirements.txt
```

## Run with the following command
```bash
python sentiment.py data/<.csv file>
```
Replace <.csv file> with the name of the dataset.

> Own data not supported yet.

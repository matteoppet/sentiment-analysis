# Sentiment analysis
This sentiment analysis program is an AI that can recognize if your comment is positive or negative. It offers two different algorithms for analysis: Logistic Regression and Sequential Neural Network.

The neural network model is already saved and will be updated if changes are made. However, if the program doesn't find the model, it will train the neural network, which may take a few minutes before you can enter your comment. If the model already exists, you will be redirected directly to the comment input prompt.

> The "sentiment_model.h5" model file must be at the same level as the "sentiment.py" file

## Demo
![Demo photo](https://github.com/matteoppet/sentiment-analysis/blob/master/photo/demo.png?raw=true)
Algorithm used in this image: [Sequential Neural Network](https://keras.io/guides/sequential_model/)

## Installation
To use this program, install the required libraries using the package manager [pip](https://pip.pypa.io/en/stable/). Run the following command:

```bash
pip install -r requirements.txt
```

## Run with the following command
```bash
python sentiment.py data/<.csv file>
```
Replace <.csv file> with the name of your dataset file. Please note that currently, only preexisting dataset is supported.

> Own data not supported yet.


## About Dataset
Dataset by [Ulrik Thyge Pedersen](https://www.kaggle.com/datasets/ulrikthygepedersen/rotten-tomatoes-reviews), released under the License: [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
The dataset used in this project is provided by [Ulrik Thyge Pedersen](https://www.kaggle.com/datasets/ulrikthygepedersen/rotten-tomatoes-reviews) and is released under the [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. If any preprocessing or modifications have been made to the original dataset, they are mentioned in the project.
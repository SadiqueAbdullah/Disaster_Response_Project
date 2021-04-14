import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import os
import sqlite3
import pickle

#Importing additional libraries for ML
import nltk
nltk.download(['wordnet', 'punkt', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """ 
    This function is to take extract data from sqlite3 database and 
    df: The dataframe containing the table from sqlite database
    X: Predictor Variable. Here it's the Text messages
    y: Target Variable. Here it's Category of messages out of 36 categories
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql_query('SELECT * FROM disaster_message_table', conn)
    X = df.message.values
    y = df.iloc[:, 4:]
    y = y.astype(str)
    
    return X, y

def tokenize(text):
    """
    This function is to process the received messages(text) into usable form.
    it normalizes, tokenize, removes stop-words, lemmatizes, and finally stores clean tokenized word in a list
    """
    
    #Text Normalization
    text= re.sub(r"[^a-zA-Z0_9]"," ", text)
    
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # Removing stop words
    stop = stopwords.words("english")
    tokens= [w for w in tokens if w not in stop]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # build pipeline
    pipeline_rf = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Creating Grid search parameters for the first model with RandomForest Classifier
    params_rf = {
            'tfidf__use_idf': (True, False),
            'clf__estimator__n_estimators': [10, 50, 100]
    }
    
    cv_rf = GridSearchCV(pipeline_rf, param_grid = params_rf)
    return cv_rf


def evaluate_model(model, X_test, Y_test):
    """
    Function: It evaluate the model and prints the  precision, recall and f1 score for each output category of the dataset.
    Args:
    model: the classification model
    X_test: Target variable of the test data
    Y_test: Target value of test data
    """
    y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """
    Function: Extract and save a pickle file of the model
    Args:
    model: the classification model
    model_filepath (str): the path of pickle file
    """

    with open(model_filepath, 'wb') as d:
        pickle.dump(model, d)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y= load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
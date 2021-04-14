# Disaster_Response_Project
This is Project is related to receiving tons of messages during a disaster and then classifying it to different categories to make best use of disaster management resources including first responders.
In the Project, I'll use a data set containing real messages that were sent during disaster events. Machine learning pipeline will be created to categorize these events so that messages can be sent to an appropriate disaster relief agency.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

# About the Data:
The data in consideration includes three 2 files, named as;

### disaster_messages.csv: 
It is a data containing messages asscociated with a disaster event. 
### disaster_categories.csv: 
This file contains a pre-work which has disaster messages charcterized into various categories.

# Libraries used:
numpy, pandas,re and seaborn etc. to clean, gather and visualize the data.
sqlalchemy, sqlite3 for SQL database manangement
nltk for Text Normalization, Tokenization, Stop word removal, Lemmatization etc.
scikit learn library to build our predictive models.

# Models used:
RandomForestClassifier
MultinomialNB
KNeighborsClassifier

These all classifiers were used with MultiOutputClassifier and later Models were tuned using GridSearch Cross Validation Technique.
Chosen RandomForestClassifier for model training in pipeline due to its best accuracy out of 3 Models.

The code should run using Python versions 3.*.

# Project Structure:
The project has three elements which are:

### ETL Pipeline: 
process_data.py file contain the script to create ETL pipline which does the following:
Loads the messages and categories datasets 
Merges the two datasets
Cleans the data
Stores it in a SQLite database in the form of a Table
### ML Pipeline:
train_classifier.py file contain the script to create ML pipline which does the following:
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Publish the results on the test set
Exports the final model as a pickle file
### Flask Web App: 
the web app enables the user to enter a message, and it higlights the categories of the message.
The web app also contains some visualizations that describe the data.

# File Structure and Description:

- README.md: read me file
- ETL_Pipeline_Preparation.ipynb: contains ETL pipeline preparation code
- ML_Pipeline_Preparation.ipynb: contains ML pipeline preparation code
- workspace
	- \app
		- run.py: flask file to run the app
	- \templates
		- master.html: web application main page
		- go.html: web page search result 
	- \data
		- disaster_categories.csv: categories dataset
		- disaster_messages.csv: messages dataset
		- DisasterResponse.db: disaster response database
		- process_data.py: ETL process code file
	- \models
		- train_classifier.py: classification code

# Instructions for execution
To execute the app follow the steps below

Run the following commands in the project's root directory to set up database and model.

### To run ETL pipeline that cleans data and stores in database 
'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
### To run ML pipeline that trains classifier and saves 
'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
### To run your web app. 
'python run.py'

Go to http://0.0.0.0:3001/

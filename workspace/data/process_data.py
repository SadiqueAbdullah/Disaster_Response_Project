import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
          Function:
          load data from two csv file and then merge them
          Args:
          messages_filepath (str): the file path of messages csv file
          categories_filepath (str): the file path of categories csv file
          Return:
          df (DataFrame): A dataframe of messages and categories after getting merged
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    
    
    
    """
      Function:
      To clean the Dataframe df
      Args:
      df (DataFrame): A dataframe of messages and categories which needs to be cleaned
      Return:
      df (DataFrame): A cleaned dataframe with messages and categories
      """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand= True)
    
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything up to the second to last character of each string with slicing
    
    header_row=0
    categories.columns= categories.iloc[header_row]
    
    #Refining Header text using str.split method
    categories.columns=categories.columns.str.split("-").str[0]
    
    #Convert category values to just numbers 0 or 1
    # set each value to be the last character of the string
    # convert column from string to numeric
    
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1]  
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df= df.drop(['categories'],axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1, join= 'inner')
    
    # drop duplicates
       
    df = df.drop_duplicates(keep='first')
    #dropping the rows with Nan values
    df= df.dropna(axis=0)
    
    # Lets drop the entries at 'related' Columns which has '2'

    df = df[df['related']!=2]
    return df

def save_data(df, database_filename):
    
    """
       Function:
       Save the Dataframe df in a sqlite3 database
       Args:
       df (DataFrame): A dataframe containing messages and categories
       database_filename (str): The file name of the database
       """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages_table', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
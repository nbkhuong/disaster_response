import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Reads two CSV files, merges them on 'id', splits the 'categories' column into individual category columns, 
    cleans up category names, and returns the resulting dataframe.

    Input:
        messages_filepath (str): Path to the messages CSV file.
        categories_filepath (str): Path to the categories CSV file.

    Returns:
        pd.DataFrame: Merged dataframe with messages and cleaned category columns.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='outer')

    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[0].str.replace(r"-\d+$", '', regex=True)
    category_colnames = row.to_list()
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str.replace(r"^[a-z_A-Z]+-", '', regex=True)

    df.drop(columns='categories', inplace=True)

    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    """
    Remove duplicate rows from a dataframe.

    Input:
        df (pd.DataFrame): The dataframe from which duplicates will be removed.

    Returns:
        pd.DataFrame: The dataframe with duplicate rows removed.
    """
     
    df = df.drop_duplicates()
    df.replace(2, 1, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the dataframe to an SQLite database.

    Input:
        df (pd.DataFrame): The dataframe to be saved.
        database_filename (str): The filename of the SQLite database.

    Returns:
        None
    """
     
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')  


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
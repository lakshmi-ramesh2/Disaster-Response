import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Function to load the messages and categories data set into dataframes that can be cleaned
    Args:
        messages_filepath: Path of messages file 
        categories_filepath: Path of categories file
    Returns:
        df: Pandas dataframe containing the messages and categories datasets combined
    """
    # load messages dataset and categories dataset
    messages = pd.read_csv(messages_filepath)  
    categories = pd.read_csv('categories.csv')
    # merge messages and categories dataframes into a single dataframe - df
    df = pd.merge(messages, categories, how='left', on=['id'])
    return df

def clean_data(df):
    """ Function to clean the dataframe by creating different category columns and removing duplicates
    Args:
        df: pandas dataframe containing the messages and categories dataset
    Returns:
        df: cleaned dataframe
    
    """
    # create a new dataframe called categories of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # Extract column names based on first row
    category_colnames = row.apply(lambda x: x[0:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category columns to numbers - 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
       categories[column] = categories[column].astype(int)
    
    # Replace categories column in df with new category columns, drop original categories column
    df = df.drop('categories',axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # Drop duplicate rows
    df = df[df.duplicated() == False]
    
    # Return clean dataframe
    return df

def save_data(df, database_filename):
    """ Function to save the data from a pandas dataframe to a sqlite database
    Args:
        df: dataframe containing the dataset to be saved to the database
        database_filename: name of the database to be created
    Returns:
        None
    """
    db_name = database_filename.split('.')[0]
    db_path = 'sqlite:///' + database_filename
    engine = create_engine(db_path)
    df.to_sql(db_name, engine, index=False)


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
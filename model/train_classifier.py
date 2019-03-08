import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """ Function to load the data from the database into a dataframe 
    Args:
        database_filepath: Path of database file 
    Returns:
        X: dataframe object containing the messages
        Y: dataframe object containing categories 
        category_names: Column names for each category
    """
    # read from database into a dataframe
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('cleaned_messages', engine)
    # create X, Y and category_names
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """ Function to tokenize any given text
    Args:
        text: Text to be tokenized
    Returns:
        clean_tokens: List of clean tokens after tokenization and cleansing
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # clean tokens by lemmatizing, making all tokens lowercase and stripping any whitespace
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Function to build pipeline with GridSearch parameters
    Args:
        None
    Returns:
        cv: GridSearchCV object for fitting
    """
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
         
    # specify parameters for GridSearch
    parameters = {        
        'vect__max_df': (0.5, 0.75),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Function to evaluate model and print out various scores
    Args:
        model: Model object to be used for prediction
        X_test: Test features for X
        Y_test: Test features for Y
        category_names: Column names for all categories
    Returns:
        None
    """
    Y_pred = model.predict(X_test)
    # Print accuracy, precision, recall and f1 scores for each feature
    for i, col in enumerate(category_names.values):
        print("Scores for: ", category_names[i])
        print("Accuracy: {:.4f}\t\t% Precision: {:.4f}\t\t% Recall: {:.4f}\t\t% F1_score: {:.4f}".format(
            accuracy_score(Y_test.loc[:, col], Y_pred[:, i]),
            precision_score(Y_test.loc[:, col], Y_pred[:, i], average='weighted'),
            recall_score(Y_test.loc[:, col], Y_pred[:, i], average='weighted'),
            f1_score(Y_test.loc[:, col], Y_pred[:, i], average='weighted')
        ))


def save_model(model, model_filepath):
    """ Function to save the model into a pickle file for future use
    Args:
        model: Model object to be saved
        model_filepath: Path of pickle file
    Returns:
        None
    """
    # save the model to create pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
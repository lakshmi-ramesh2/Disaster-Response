# Disaster Response Pipeline Project


### Table of Contents

1. [Libraries Used](#libraries)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)

## Libraries Used <a name="libraries"></a>

This project was implemented using the Anaconda distribution of Python 3.0. The most important python libraries that were used are:

1. NumPy and Pandas - for data cleansing and wrangling
2. NLTK - for natural language processing
3. Sklearn - for modeling

## Project Motivations<a name="motivation"></a>

For this project, here are the key tasks that were accomplished:

1. Create an ETL pipeline that reads in disaster response messages from csv files and does data cleansing
2. Create an ML pipeline that includes Natural Language Processing of the message data and implementing a classification model
3. Visualize the results in a web app and embed the model into the app for prediction


## File Descriptions <a name="files"></a>

There are 2 jupyter notebooks that contain the ETL and ML pipelines.

1. ETL Pipeline Preparation.ipynb
2. ML Pipeline Preparation.ipynb

In addition, there are 3 main folders:

1. data - containing the raw messages and categories dataset as well as a python script process_data.py that reads in the data and does the cleansing and finally stores the cleaned data in a database table called cleaned_messages

2. model - reads the dataset from the database, does NLP tasks and implements a classification model

3. app - files to run the web app that embeds the dataset and the model.

To run the files, follow the below instructions:

1. To run the ETL pipeline:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To run the ML pipeline:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. To run the web app, go to the app folder:
    `python run.py`



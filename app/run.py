import json
import plotly
import pandas as pd
import re
import nltk
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('cleaned_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)
    
    # Extract category names and counts for each category
    cat_names = df.iloc[:,4:].columns.values
    cat_counts = []
    for c in cat_names: 
        cat_counts.append(df[c].sum())
    
    # Extract top 10 keywords for 100 messages under the news genre
    df_news = df[df['genre'] == 'news']
    df_news = df_news[:100]
    news_keywords = []
    for index, row in df_news.iterrows():
        news_tokens = tokenize(row['message'])
        for t in news_tokens:
          news_keywords.append(t)

    # Remove stop words from news_keywords
    news_keywords = [word for word in news_keywords if word not in stopwords.words('english')]
    
    # Count the frequency of words and create the top 10 lists
    from collections import Counter
    words_to_count = (word for word in news_keywords)
    c = Counter(words_to_count)
    top_words = []
    top_word_counts = []
    for t in c.most_common(10):
        top_words.append(t[0])
        top_word_counts.append(t[1])
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Category Count"
                },
                'xaxis': {
                    'title': "Category Name"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_word_counts
                )
            ],

            'layout': {
                'title': 'Top Words in News Messages',
                'yaxis': {
                    'title': "Top Word Count"
                },
                'xaxis': {
                    'title': "Top Words"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sqlite3

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages_table', engine)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    rel_count = df.groupby('related').count()['message']
    rel_per = round(100*rel_count/rel_count.sum(), 2)
    rel = list(rel_count.index)
    cat_num = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    cat_num = cat_num.sort_values(ascending = False)
    cat = list(cat_num.index)
    direct_count = df.groupby('direct_report').count()['message']
    direct = list(direct_count.index)

        
    # create visuals
    
    graphs = [
        # First graph as visaul to Dashboard    
        {
            "data": [
              {
                "type": "bar",
                "x": cat,
                "y": cat_num,
                "marker": {
                  "color": 'blue'}
                }
            ],
            "layout": {
              "title": "Count of Messages by Category",
              'yaxis': {
                  'title': "Count"
              },
              'xaxis': {
                  'title': "Categories"
              },
              'barmode': 'group'
            }
        },
        
        # Second Graph for Dashboard
        
        {
            "data": [
              {
                "type": "bar",
                "x": direct,
                "y": direct_count,
                "marker": {
                  "color": 'orange'}
                }
            ],
            "layout": {
              "title": "Count of Messages received directly or indirectly",
              'yaxis': {
                  'title': "Count"
              },
              'xaxis': {
                  'title': "direct reports (Yes/No)"
              },
              'barmode': 'group'
            }
        },
        # Third graph for Dashboard
        {
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "hole": 0.8,
                "name": "Related",
                "pull": 0,
                "domain": {
                  "x": rel_per,
                  "y": rel
                },
                "marker": {
                  "colors": [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": rel,
                "values": rel_per
              }
            ],
            "layout": {
              "title": "Percentage of Messages by Related categories"
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
from flask import current_app as app
from flask import render_template, redirect, url_for, jsonify, request, abort
from .utils import LoadArticles


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/topics', methods=['POST'])
def compute_topics():
    load_articles = LoadArticles()
    # df = load_articles.load_articles(datetime.date(2023, 7, 14), datetime.date(2023, 7, 21))
    print(request.form['startDate'])
    print(request.form['endDate'])
    print(load_articles.engine)
    return 'nice'
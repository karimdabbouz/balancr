from flask import current_app as app
from flask import render_template, redirect, url_for, jsonify, request, abort
from .models import db, table_list, zeitArticles
from .extensions import scheduler, cache
from .utils import compute_lda_topics


@app.route('/')
def home():
    return 'balancr-api'


@app.route('/api/lda-topics')
def lda_topics():
    '''
    Allows to explicitly recompute LDA topics.
    '''
    lda_topics = compute_lda_topics()
    return jsonify(lda_topics)


@app.route('/test-db')
def test_db():
    print(type(zeitArticles))
    # entry = db.session.query(zeitArticles).first()
    return "yeah"
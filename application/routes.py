from flask import current_app as app
from flask import render_template, redirect, url_for, jsonify, request, abort


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/topics', methods=['POST'])
def compute_topics():
    print(request.form['startDate'])
    print(request.form['endDate'])
    return 'nice'
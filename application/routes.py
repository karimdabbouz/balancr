from flask import current_app as app
from flask import render_template, redirect, url_for, jsonify, request, abort


@app.route('/')
def home():
    return 'balancr-api'
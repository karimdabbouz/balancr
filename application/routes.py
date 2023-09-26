from flask import current_app as app
from flask import render_template, redirect, url_for, jsonify, request, abort
from .utils import LoadArticles, FilterArticles, BERTTopicModeling, ProcessTopics
import datetime, os


@app.route('/')
def home():
    os.environ.get('SQLALCHEMY_DATABASE_URI')
    return render_template('index.html')


@app.route('/topics', methods=['POST'])
def compute_topics():
    load_articles = LoadArticles()
    start_date = datetime.datetime.strptime(request.form.getlist('startDate')[0], '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(request.form.getlist('endDate')[0], '%Y-%m-%d').date()
    keep_headline = True if len(request.form.getlist('keepHeadline')) > 0 else False
    keep_teaser = True if len(request.form.getlist('keepTeaser')) > 0 else False
    keep_body = True if len(request.form.getlist('keepBody')) > 0 else False
    df = load_articles.load_articles(start_date, end_date)
    if len(df) > 0:
        docs = []
        filter_articles = FilterArticles(keep_headline, keep_teaser, keep_body)
        filtered_df = filter_articles.filter_articles(df)
        topic_model = BERTTopicModeling('paraphrase-multilingual-MiniLM-L12-v2')
        if keep_headline == True:
            docs.append(list(filtered_df['headline']))
        if keep_teaser == True:
            docs.append(list(filtered_df['teaser']))
        if keep_body == True:
            docs.append(list(filtered_df['body']))
        final_docs = [' '.join(items) for items in zip(*docs)]
        topics, probabilities = topic_model.fit_transform(final_docs)
        process_topics = ProcessTopics(topic_model.topic_model, final_docs, filtered_df)
        topics_table = topic_model.topic_model.get_topic_info().drop('Representative_Docs', axis=1).iloc[1:]
        top_topics_plot = process_topics.visualize_num_docs_per_topic()
        process_topics.compute_num_articles_per_medium()
        num_docs_per_medium_plots = process_topics.visualize_num_docs_per_medium()
        topic_model.topic_model.save(f'./saved_models/{start_date}_{end_date}-{keep_headline}|{keep_teaser}|{keep_body}.pkl')
        return render_template('topics.html',
                                topics_table=topics_table.to_html(classes='table table-striped table-bordered', escape=False),
                                top_topics_plot=top_topics_plot,
                                num_docs_per_medium_plots=num_docs_per_medium_plots,
                                start_date=request.form.getlist('startDate')[0],
                                end_date=request.form.getlist('endDate')[0],
                                num_articles_published=len(df),
                                headline_teaser_body=[keep_headline, keep_teaser, keep_body],
                                num_articles_used_in_modeling=len(final_docs),
                                sources=filtered_df['medium'].unique())
    else:
        return 'there aren\t any articles in the db for the time span you entered'
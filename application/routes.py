from flask import current_app as app
from flask import render_template, redirect, url_for, jsonify, request, abort, session
from .utils import LoadArticles, FilterArticles, BERTTopicModeling, ProcessTopics, save_results, load_results
import datetime, os


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/topic/<topic_id>')
def topic_page(topic_id):
    articles = []
    load_articles = LoadArticles()
    consolidated_data = {}
    article_ids = session.get('articles')[topic_id]
    for item in article_ids:
        medium = item['medium']
        id_value = item['id']
        if medium in consolidated_data:
            consolidated_data[medium].append(id_value)
        else:
            consolidated_data[medium] = [id_value]
    consolidated_list = [{'medium': medium, 'ids': ids} for medium, ids in consolidated_data.items()]
    for entry in consolidated_list:
        response = load_articles.get_articles_for_ids(f'{entry["medium"]}_articles', entry['ids'])
        articles.extend(response)
    return render_template('topic_page.html',
        topic_id=topic_id,
        articles=articles)


@app.route('/topics_overview', methods=['POST'])
def compute_topics():
    start_date = datetime.datetime.strptime(request.form.getlist('startDate')[0], '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(request.form.getlist('endDate')[0], '%Y-%m-%d').date()
    keep_headline = True if len(request.form.getlist('keepHeadline')) > 0 else False
    keep_teaser = True if len(request.form.getlist('keepTeaser')) > 0 else False
    keep_body = True if len(request.form.getlist('keepBody')) > 0 else False
    load_articles = LoadArticles()
    all_articles_df = load_articles.load_articles(start_date, end_date)
    try:
        # try loading existing model
        filtered_df, labels_and_summaries_df, topic_model = load_results(start_date, end_date, keep_headline, keep_teaser, keep_body)
        docs = []
        if keep_headline == True:
            docs.append(list(filtered_df['headline']))
        if keep_teaser == True:
            docs.append(list(filtered_df['teaser']))
        if keep_body == True:
            docs.append(list(filtered_df['body']))
        final_docs = [' '.join(items) for items in zip(*docs)]
        process_topics = ProcessTopics(topic_model, final_docs, filtered_df, './service-account-key.json')
        topics_table = topic_model.get_topic_info().drop('Representative_Docs', axis=1).iloc[1:]
        top_topics_plot = process_topics.visualize_num_docs_per_topic()
        process_topics.compute_num_articles_per_medium()
        num_docs_per_medium_plots = process_topics.visualize_num_docs_per_medium()
        session_data = process_topics.build_article_data()
        session['articles'] = session_data
        return render_template('topics_overview.html',
                        topics_table=topics_table.to_html(classes='table table-striped table-bordered', escape=False),
                        top_topics_plot=top_topics_plot,
                        num_docs_per_medium_plots=num_docs_per_medium_plots,
                        start_date=request.form.getlist('startDate')[0],
                        end_date=request.form.getlist('endDate')[0],
                        num_articles_published=len(all_articles_df),
                        headline_teaser_body=[keep_headline, keep_teaser, keep_body],
                        num_articles_used_in_modeling=len(final_docs),
                        sources=filtered_df['medium'].unique(),
                        # topic_labels=[(i, row['label']) for i, row in labels_and_summaries_df.iloc[:3].iterrows()],
                        # topic_summaries=[(i, row['label']) for i, row in labels_and_summaries_df.iloc[:3].iterrows()])
                        topic_labels=[(i, row['label']) for i, row in labels_and_summaries_df.iloc[:3].iterrows()],
                        topic_summaries=[(i, row['summary']) for i, row in labels_and_summaries_df.iloc[:3].iterrows()])
    except:
        # train model
        print('retraining')
        if len(all_articles_df) > 0:
            filter_articles = FilterArticles(keep_headline, keep_teaser, keep_body)
            filtered_df = filter_articles.filter_articles(all_articles_df)
            docs = []
            if keep_headline == True:
                docs.append(list(filtered_df['headline']))
            if keep_teaser == True:
                docs.append(list(filtered_df['teaser']))
            if keep_body == True:
                docs.append(list(filtered_df['body']))
            final_docs = [' '.join(items) for items in zip(*docs)]
            topic_model = BERTTopicModeling('paraphrase-multilingual-MiniLM-L12-v2')
            topics, probabilities = topic_model.fit_transform(final_docs)
            process_topics = ProcessTopics(topic_model.topic_model, final_docs, filtered_df, './service-account-key.json')
            topics_table = topic_model.topic_model.get_topic_info().drop('Representative_Docs', axis=1).iloc[1:]
            top_topics_plot = process_topics.visualize_num_docs_per_topic()
            process_topics.compute_num_articles_per_medium()
            process_topics.compute_llm_topic_labels()
            process_topics.compute_llm_topic_summaries()
            num_docs_per_medium_plots = process_topics.visualize_num_docs_per_medium()
            save_results(start_date, end_date, topic_model, keep_headline, keep_teaser, keep_body, process_topics, filtered_df)
            session_data = process_topics.build_article_data()
            session['articles'] = session_data     
            return render_template('topics_overview.html',
                                    topics_table=topics_table.to_html(classes='table table-striped table-bordered', escape=False),
                                    top_topics_plot=top_topics_plot,
                                    num_docs_per_medium_plots=num_docs_per_medium_plots,
                                    start_date=request.form.getlist('startDate')[0],
                                    end_date=request.form.getlist('endDate')[0],
                                    num_articles_published=len(all_articles_df),
                                    headline_teaser_body=[keep_headline, keep_teaser, keep_body],
                                    num_articles_used_in_modeling=len(final_docs),
                                    sources=filtered_df['medium'].unique(),
                                    topic_labels=process_topics.topic_labels,
                                    topic_summaries=process_topics.topic_summaries)
        else:
            return ('no articles for this time frame in database')
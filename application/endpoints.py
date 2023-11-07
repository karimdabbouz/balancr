from flask import current_app as app
from flask import request, jsonify
from flask_cors import cross_origin
import datetime, json, itertools
from .utils import LoadArticles, BERTTopicModeling, FilterArticles, load_predictions, save_predictions


@app.route('/get_topic_predictions', methods=['GET'])
@cross_origin(origins=['http://localhost:8080'], methods=['GET'])
def get_topic_predictions():
    print("get_topic_predictions() was called")
    start_date = datetime.datetime.strptime(request.args.get('start_date'), '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(request.args.get('end_date'), '%Y-%m-%d').date()
    parts_of_article = request.args.get('parts_of_article')
    sources = request.args.get('sources').split(',')
    keep_kicker = True if request.args.get('parts_of_article').split(',')[0] == 'true' else False
    keep_headline = True if request.args.get('parts_of_article').split(',')[1] == 'true' else False
    keep_teaser = True if request.args.get('parts_of_article').split(',')[2] == 'true' else False
    keep_body = True if request.args.get('parts_of_article').split(',')[3] == 'true' else False
    include_paywalled = True if request.args.get('paywall') == 'true' else False
    try:
        predictions = load_predictions(
            app.config['baseline_model_name'],
            start_date,
            end_date,
            keep_kicker,
            keep_headline,
            keep_teaser,
            keep_body,
            include_paywalled,
            sources
        )
        return jsonify(predictions)
    except:
        print("get_topic_predictions(): computing a new prediction")
        load_articles = LoadArticles()
        all_articles_df = load_articles.load_articles(start_date, end_date, True)
        if len(all_articles_df) > 0:
            filter_articles = FilterArticles(keep_kicker, keep_headline, keep_teaser, keep_body, sources)
            filtered_df = filter_articles.filter_articles(all_articles_df)
            docs = []
            if keep_kicker == True:
                docs.append(list(filtered_df['kicker']))
            if keep_headline == True:
                docs.append(list(filtered_df['headline']))
            if keep_teaser == True:
                docs.append(list(filtered_df['teaser']))
            if keep_body == True:
                docs.append(list(filtered_df['body']))
            final_docs = [' '.join(items) for items in zip(*docs)]
            baseline_model_name = app.config['baseline_model_name']
            topic_model = BERTTopicModeling('paraphrase-multilingual-MiniLM-L12-v2')
            topics, probabilities = topic_model.transform(f'./modeling_results/{baseline_model_name}', final_docs)
            predictions = save_predictions(
                topics,
                probabilities,
                filtered_df,
                app.config['baseline_model_name'],
                start_date,
                end_date,
                keep_kicker,
                keep_headline,
                keep_teaser,
                keep_body,
                include_paywalled,
                sources
            )
            return jsonify(predictions)
        else:
            raise Exception('not enough articles in database for these parameters')


@app.route('/get_baseline_model_data', methods=['GET'])
@cross_origin(origins=['http://localhost:8080'], methods=['GET'])
def get_baseline_model_data():
    with open('./modeling_results/baseline_model_data_2023-10-24.json', 'r') as json_file:
        data = json.load(json_file)
    return jsonify(data)


@app.route('/get_articles', methods=['POST'])
@cross_origin(origins=['http://localhost:8080'], methods=['POST'])
def get_articles():
    result = []
    load_articles = LoadArticles()
    data = request.get_json()
    consolidated_dict = {}
    for key, value in data:
        if key in consolidated_dict:
            consolidated_dict[key].append(value)
        else:
            consolidated_dict[key] = [value]
    for key, value in consolidated_dict.items():
        articles = load_articles.get_articles_for_ids(f'{key}_articles', value)
        result.extend(articles)
    list_of_dicts = [{
        'id': item[0],
        'medium': item[1],
        'datetime_saved': datetime.datetime.strftime(item[2], '%Y-%m-%d'),
        'date_published': datetime.datetime.strftime(item[3], '%Y-%m-%d'),
        'url': item[4],
        'headline': item[5],
        'kicker': item[6],
        'teaser': item[7],
        'body': item[8],
        'subheadlines': item[9],
        'paywall': item[10],
        'author': item[11],
        'code': item[12],
        'archive_url': item[13]
    } for item in result]
    return jsonify(list_of_dicts)

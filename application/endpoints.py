from flask import current_app as app
from flask import request, jsonify
import datetime
from .utils import LoadArticles, BERTTopicModeling, FilterArticles, load_predictions, save_predictions


@app.route('/get_topics_overview', methods=['GET'])
def get_topics_overview():
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
            raise Exception('something went wrong')



# THIS IS PROBABLY NOT NECESSARY SINCE DATA CAN BE PASSED AS A PROP IN THE FRONTEND FROM get_topics_overview()
# @app.route('/get_topic', methods=['GET'])
# def get_topic():
#     # call this when a user clicks on a given topic from a topic overview
#     # -> needs to know the model and the topic_id
#     start_date = request.args.get('start_date')
#     end_date = request.args.get('end_date')
#     sources = request.args.get('sources')

#     pass
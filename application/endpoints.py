from flask import current_app as app
from flask import request
import datetime


@app.route('/get_topics_overview', methods=['GET'])
def get_topics_overview():
    start_date = datetime.datetime.strptime(request.args.get('start_date'), '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(request.args.get('end_date'), '%Y-%m-%d').date()
    parts_of_article = request.args.get('parts_of_article')
    sources = request.args.get('sources')
    keep_kicker = True if request.args.get('parts_of_article').split(',')[0] == 'true' else False
    keep_headline = True if request.args.get('parts_of_article').split(',')[1] == 'true' else False
    keep_teaser = True if request.args.get('parts_of_article').split(',')[2] == 'true' else False
    keep_body = True if request.args.get('parts_of_article').split(',')[3] == 'true' else False
    paywall = True if request.args.get('paywall') == 'true' else False
    # WHAT ELSE SHOULD YOU BE ABLE TO FILTER FOR?
    # -> THIS WOULD RESULT IN MANY DIFFERENT MODELS DEPENDING ON THE SETTINGS. THERE NEEDS TO BE A BETTER WAY. PAUSING HERE...


    # load_articles = LoadArticles()
    # all_articles_df = load_articles.load_articles(start_date, end_date)


    example_result = {
        'start_date': start_date,
        'end_date': end_date,
        'total_num_articles': 666,
        'parts_of_article': [True, True, True, True],
        'sources': ['zeit', 'faz', 'tagesschau', 'welt', 'sueddeutsche'],
        'topics': {
            'topic_0': {
                'articles': [
                    {
                        'article_id': 7,
                        'source': 'zeit',
                        'headline': 'lorem ipsum headline',
                        'probability': 0.98
                    },
                    {
                        'article_id': 3,
                        'source': 'zeit',
                        'headline': 'lorem ipsum headline2',
                        'probability': 0.77
                    }
                ]
            }
        }
    }
    return example_result


# THIS IS PROBABLY NOT NECESSARY SINCE DATA CAN BE PASSED AS A PROP IN THE FRONTEND FROM get_topics_overview()
# @app.route('/get_topic', methods=['GET'])
# def get_topic():
#     # call this when a user clicks on a given topic from a topic overview
#     # -> needs to know the model and the topic_id
#     start_date = request.args.get('start_date')
#     end_date = request.args.get('end_date')
#     sources = request.args.get('sources')

#     pass
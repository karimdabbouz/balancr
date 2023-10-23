from flask import current_app as app
from flask import request


@app.route('/get_topics_overview', methods=['GET'])
def get_topics_overview():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    example_result = {
        'start_date': start_date,
        'end_date': end_date,
        'total_num_articles': 666,
        'parts_of_article': [True, True, True, True],
        'sources': ['zeit', 'faz', 'tagesschau', 'welt'],
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
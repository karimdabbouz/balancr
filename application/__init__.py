from flask import Flask, url_for
from dotenv import load_dotenv
from bertopic import BERTopic



def create_app():
    app = Flask(__name__)
    app.config.from_object('config.DevConfig')
    load_dotenv()

    baseline_model_name = 'baseline_model_2023-10-24.pkl'
    app.config['baseline_model_name'] = baseline_model_name
    
    model = BERTopic.load(f'./modeling_results/{baseline_model_name}')
    app.config['model'] = model

    with app.app_context():
        from . import routes
        from . import endpoints

    return app
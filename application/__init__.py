from flask import Flask, url_for
from dotenv import load_dotenv



def init_app():
    app = Flask(__name__)
    app.config.from_object('config.DevConfig')
    load_dotenv()


    with app.app_context():
        from . import routes
        from . import endpoints

        return app
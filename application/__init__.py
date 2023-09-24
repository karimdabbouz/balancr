from flask import Flask
from dotenv import load_dotenv
# from .extensions import db



def init_app():
    app = Flask(__name__)
    app.config.from_object('config.DevConfig')
    load_dotenv()

    # db.init_app(app)

    with app.app_context():
        from . import routes

        return app
from flask import Flask
from .extensions import db, scheduler, cache, r



def init_app():
    app = Flask(__name__)
    app.config.from_object('config.DevConfig')

    db.init_app(app)
    # scheduler.init_app(app)
    # cache.init_app(app)
    # r.init_app(app)

    with app.app_context():
        from . import routes
        from . import tasks

        # scheduler.start()

        return app
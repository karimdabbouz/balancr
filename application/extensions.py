from flask_apscheduler import APScheduler
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_redis import FlaskRedis


db = SQLAlchemy()
scheduler = APScheduler()
cache = Cache()
r = FlaskRedis(decode_responses=True)
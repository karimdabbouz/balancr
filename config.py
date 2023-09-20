import os


class Config:
    ''' Base config '''
    SECRET_KEY = os.environ.get('SECRET_KEY')
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'

    # Articles Table
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')

    # User Table
    SQLALCHEMY_BINDS = os.environ.get('SQLALCHEMY_BINDS')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Redis Config
    REDIS_URL = os.environ.get('REDIS_URL')


class DevConfig(Config):
    ''' Configurations for development '''
    FLASK_ENV  = 'development'
    DEBUG = True
    TESTING = True
    SQLALCHEMY_ECHO = True


class ProdConfig(Config):
    ''' Configurations for production '''
    FLASK_ENV = 'production'
    DEBUG = False
    TESTING = False
    SQLALCHEMY_ECHO = False

import os


class Config:
    ''' Base config '''
    SECRET_KEY = os.environ.get('SECRET_KEY')
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    # Articles Table
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')

    # User Table
    SQLALCHEMY_BINDS = os.environ.get('SQLALCHEMY_BINDS')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

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

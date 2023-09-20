from . import db


class fazArticles(db.Model):
    __tablename__ = 'faz_articles'
    id = db.Column(db.Integer, primary_key=True)
    medium = db.Column(db.String)
    datetime_saved = db.Column(db.DateTime)
    date_published = db.Column(db.DateTime)
    url = db.Column(db.String)
    headline = db.Column(db.String)
    kicker = db.Column(db.String)
    teaser = db.Column(db.String)
    body = db.Column(db.String)
    subheadlines = db.Column(db.String)
    paywall = db.Column(db.Boolean)
    author = db.Column(db.String)
    code = db.Column(db.String)
    archive_url = db.Column(db.String)


class sueddeutscheArticles(db.Model):
    __tablename__ = 'sueddeutsche_articles'
    id = db.Column(db.Integer, primary_key=True)
    medium = db.Column(db.String)
    datetime_saved = db.Column(db.DateTime)
    date_published = db.Column(db.DateTime)
    url = db.Column(db.String)
    headline = db.Column(db.String)
    kicker = db.Column(db.String)
    teaser = db.Column(db.String)
    body = db.Column(db.String)
    subheadlines = db.Column(db.String)
    paywall = db.Column(db.Boolean)
    author = db.Column(db.String)
    code = db.Column(db.String)
    archive_url = db.Column(db.String)


class tagesschauArticles(db.Model):
    __tablename__ = 'tagesschau_articles'
    id = db.Column(db.Integer, primary_key=True)
    medium = db.Column(db.String)
    datetime_saved = db.Column(db.DateTime)
    date_published = db.Column(db.DateTime)
    url = db.Column(db.String)
    headline = db.Column(db.String)
    kicker = db.Column(db.String)
    teaser = db.Column(db.String)
    body = db.Column(db.String)
    subheadlines = db.Column(db.String)
    paywall = db.Column(db.Boolean)
    author = db.Column(db.String)
    code = db.Column(db.String)
    archive_url = db.Column(db.String)


class weltArticles(db.Model):
    __tablename__ = 'welt_articles'
    id = db.Column(db.Integer, primary_key=True)
    medium = db.Column(db.String)
    datetime_saved = db.Column(db.DateTime)
    date_published = db.Column(db.DateTime)
    url = db.Column(db.String)
    headline = db.Column(db.String)
    kicker = db.Column(db.String)
    teaser = db.Column(db.String)
    body = db.Column(db.String)
    subheadlines = db.Column(db.String)
    paywall = db.Column(db.Boolean)
    author = db.Column(db.String)
    code = db.Column(db.String)
    archive_url = db.Column(db.String)


class zeitArticles(db.Model):
    __tablename__ = 'zeit_articles'
    id = db.Column(db.Integer, primary_key=True)
    medium = db.Column(db.String)
    datetime_saved = db.Column(db.DateTime)
    date_published = db.Column(db.DateTime)
    url = db.Column(db.String)
    headline = db.Column(db.String)
    kicker = db.Column(db.String)
    teaser = db.Column(db.String)
    body = db.Column(db.String)
    subheadlines = db.Column(db.String)
    paywall = db.Column(db.Boolean)
    author = db.Column(db.String)
    code = db.Column(db.String)
    archive_url = db.Column(db.String)

table_list = [fazArticles, sueddeutscheArticles, tagesschauArticles, weltArticles, zeitArticles]
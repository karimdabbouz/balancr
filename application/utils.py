from datetime import datetime, timedelta, timezone
from .models import db, table_list
from flask import current_app as app
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from textblob_de import TextBlobDE
from gensim import corpora, models
from .extensions import r
import nltk, collections, random



def compute_lda_topics(table_list=table_list):
    # 1) For each table in a list of tables: Load articles for the last 6 hours
    docs = []
    for x in table_list:
        entries = db.session.query(x).filter((x.date_published >= datetime.utcnow() - timedelta(hours=6)) & (x.date_published < datetime.utcnow())).all()
        docs += [{'medium': x.medium, 'url': x.url, 'kicker': x.kicker, 'headline': x.headline, 'teaser': x.teaser, 'body': x.body.replace('„', '').replace('“', '').replace(';', '').replace('(', '').replace(')', '').replace('–', '')} for x in entries if x.body != None]

    # Tokenize and keep only nouns
    tokens_only_nouns = [[y[0] for y in TextBlobDE(x['body']).tags if (y[1] == 'NN' or y[1] == 'NNS' or y[1] == 'NNP' or y[1] == 'NNPS')] for x in docs]
    
    # Build list of docs with term counts for each document
    dct = corpora.Dictionary(tokens_only_nouns)
    corpus = [dct.doc2bow(x) for x in tokens_only_nouns] # BoW-representation
    term_counts = [[(dct[id], count) for id, count in doc] for doc in corpus] # Map terms to corpus

    # Dimensionality Reduction
    lda = models.LdaModel(corpus, num_topics=5)

    # Build list of list of topic terms and their scores in each topic
    topic_terms = [[(dct[x[0]], x[1]) for x in topic] for topic in [lda.get_topic_terms(doc) for doc in range(lda.num_topics)]]

    # Build list with most important (most occuring) terms for each document
    most_important_terms = [[x[0] for x in entry] for entry in [sorted(doc, key=lambda x: x[1], reverse=True)[:5] for doc in term_counts]]
    
    # Build overview per article
    article_overview = []
    for i, v in enumerate(lda.get_document_topics(corpus)):
        article = {
            'index': i,
            'topic': sorted(v, key=lambda x: x[1], reverse=True)[0][0],
            'url': str(docs[i]['url']),
            'kicker': str(docs[i]['kicker']),
            'headline': str(docs[i]['headline']),
            'teaser': str(docs[i]['teaser']),
            'body': str(docs[i]['body']),
            'medium': str(docs[i]['medium']),
            'most_important_terms': str(most_important_terms[i])
        }
        article_overview.append(article)

    # Bring into format to get a random collections from
    topic_list = collections.Counter([x['topic'] for x in article_overview]).keys()
    topic_medium_tuples = []
    for topic in topic_list:
        for i, v in enumerate(article_overview):
            if v['topic'] == topic:
                topic_medium_tuples.append((v['index'], v['topic'], v['medium']))
    
    # Build random collection of articles for each topic and save to Redis
    r.flushall()
    for topic in topic_list:
        random_choice = random.choices([x for x in topic_medium_tuples if x[1] == topic], k=4)
        for article in random_choice:
            r.hset('lda_topics_article{}'.format(article[0]), mapping=article_overview[article[0]])

    return article_overview
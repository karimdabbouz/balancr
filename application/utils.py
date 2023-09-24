import os
from sqlalchemy import create_engine, MetaData, Table, select, inspect, and_
import pandas as pd
import plotly.express as px

from nltk.corpus import stopwords
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer



class LoadArticles():
    def __init__(self):
        self.engine = create_engine(os.environ.get('SQLALCHEMY_DATABASE_URI'))
        self.metadata = MetaData()
        self.tables = inspect(self.engine).get_table_names()


    def load_articles(self, start_date, end_date):
        articles = []
        for table in self.tables:
            table_data = Table(table, self.metadata, autoload_with=self.engine)
            query = select(table_data).where(and_(table_data.c.date_published >= start_date, table_data.c.date_published <= end_date))
            with self.engine.connect() as connection:
                response = connection.execute(query)
                articles.extend(response)
        return pd.DataFrame(articles)


class FilterArticles():
    def __init__(self, headline, teaser, body):
        self.headline = headline
        self.teaser = teaser
        self.body = body

    def filter_articles(self, df):
        common_indices = []
        if self.headline == True:
            filtered_df = df[(df['headline'].notna()) & (df['headline'] != '')]
            headline_indices = list(filtered_df.index)
            common_indices.append(set(headline_indices))
        if self.teaser == True:
            filtered_df = df[(df['teaser'].notna()) & (df['teaser'] != '')]
            teaser_indices = list(filtered_df.index)
            common_indices.append(set(teaser_indices))
        if self.body == True:
            filtered_df = df[(df['body'].notna()) & (df['body'] != '')]
            body_indices = list(filtered_df.index)
            common_indices.append(set(body_indices))
            
        if common_indices:
            common_indices = list(set.intersection(*common_indices))
            result_df = df.iloc[common_indices]
            result_df['full_content'] = result_df['headline'] + ' ' + result_df['teaser'] + ' ' + result_df['body']

        result_df = df.iloc[common_indices]
        return result_df


class BERTTopicModeling():
    def __init__(self, embedding_model):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        self.hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        self.german_stopwords = stopwords.words('german')
        self.vectorizer_model = CountVectorizer(stop_words=self.german_stopwords)
        self.ctfidf_model = ClassTfidfTransformer()
        self.representation_model = KeyBERTInspired()
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model
        )

    def fit_transform(self, docs):
        topics, probabilities = self.topic_model.fit_transform(docs)
        return topics, probabilities


class ProcessTopics():
    def __init__(self, topic_model, docs, docs_df):
        self.topic_model = topic_model
        self.docs = docs
        self.docs_df = docs_df
        self.top_topics = pd.DataFrame([{'topic': row['Topic'], 'representation': row['Representation'], 'num_docs': row['Count']} for i, row in self.topic_model.get_topic_info().iloc[1:11].iterrows()])
        self.documents_df = topic_model.get_document_info(docs)
        self.num_articles_per_medium = {}

    def compute_num_articles_per_medium(self):
        media_list = []
        media_counts = {}
        for i, row in self.documents_df.iterrows():
            media_list.append(self.docs_df.iloc[i].medium)
        self.documents_df['medium'] = media_list
        for i, row in self.top_topics.iterrows():
            matches = self.documents_df[self.documents_df['Topic'] == row['topic']].sort_values(by='Probability')
            medium_counts = matches['medium'].value_counts().to_dict()
            media_counts[row['topic']] = medium_counts
        self.num_articles_per_medium = media_counts
        return media_counts

    def visualize_num_docs_per_topic(self):
        self.top_topics['representation'] = [' '.join([x for x in sublist[:5]]) for sublist in self.top_topics['representation']]
        top_topics_fig = px.bar(self.top_topics, x='representation', y='num_docs', color_discrete_sequence=['#3C4856'])
        top_topics_fig.update_layout(title='Zehn wichtigste Themen nach Anzahl Artikel insgesamt')
        return top_topics_fig.to_html(full_html=False)

    def visualize_num_docs_per_medium(self):
        results = []
        for i, row in self.top_topics.iloc[:3].iterrows():
            df = pd.DataFrame(self.num_articles_per_medium[row['topic']].items(), columns=['medium', 'num_articles'])
            num_articles_per_medium_fig = px.bar(df, x='medium', y='num_articles', color_discrete_sequence=['#3C4856'])
            topic_description = ''.join([sublist for sublist in row['representation']])
            num_articles_per_medium_fig.update_layout(title=f'Anzahl Artikel je Medium zum Thema: {topic_description}')
            results.append(num_articles_per_medium_fig.to_html(full_html=False))
        return results
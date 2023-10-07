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

from vertexai.language_models import TextGenerationModel

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

    def get_articles_for_ids(self, tablename, id_list):
        articles = []
        table_data = Table(tablename, self.metadata, autoload_with=self.engine)
        query = select(table_data).where(table_data.c.id.in_(id_list))
        with self.engine.connect() as connection:
            response = connection.execute(query)
        return response.fetchall()


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
        self.representation_model =  KeyBERTInspired()
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
    def __init__(self, topic_model, docs, docs_df, service_account_key):
        self.topic_model = topic_model
        self.docs = docs
        self.docs_df = docs_df
        self.top_topics = pd.DataFrame([{'topic': row['Topic'], 'representation': row['Representation'], 'num_docs': row['Count']} for i, row in self.topic_model.get_topic_info().iloc[1:11].iterrows()])
        self.documents_df = topic_model.get_document_info(docs, df=docs_df)
        self.num_articles_per_medium = {}
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key
        self.llm_model = TextGenerationModel.from_pretrained('text-bison')
        self.topic_labels = [(0, 'foo')]
        self.topic_summaries = [(0, 'foo')]

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

    def compute_llm_topic_labels(self):
        parameters = {
            'temperature': 0.2, # increase if answers are too generic
            'max_output_tokens': 200
        }
        base_prompt = 'Ich habe Themencluster aus Nachrichtenartikeln modelliert. Der folgende Text ist ein Auszug aus drei Beispielartikeln eines Clusters. Bitte gib mir einen Titel für das Thema, von dem die Artikel handeln. Bitte nenne nur den Titel und nutze maximal 5 Wörter: '
        labels = []
        for i in range(10):
            try:
                articles = [x[:10000] for x in self.topic_model.get_representative_docs(i)]
                prompt = base_prompt + ' '.join(articles)
                topic_label = self.llm_model.predict(prompt, **parameters)
                labels.append((i, topic_label))
            except:
                summaries.append((i, 'foo'))
        self.topic_labels = labels

    def compute_llm_topic_summaries(self):
        parameters = {
            'temperature': 0.2, # increase if answers are too generic
            'max_output_tokens': 500
        }
        base_prompt = 'Worum geht es in folgendem Text? Bitte fasse in wenigen Sätzen zusammen: '
        summaries = []
        for i in range(10):
            try:
                articles = [x[:10000] for x in self.topic_model.get_representative_docs(i)]
                prompt = base_prompt + ' '.join(articles)
                topic_summary = self.llm_model.predict(prompt, **parameters)
                summaries.append((i, topic_summary))
            except:
                summaries.append((i, 'foo'))
        self.topic_summaries = summaries

    def build_article_data(self):
        result_dict = {}
        for i in range(10):
            articles_list = []
            articles = self.documents_df[self.documents_df['Topic'] == i].sort_values(by='Probability', ascending=False).iloc[:30]
            for index, row in articles.iterrows():
                data = {
                    'id': row['id'],
                    'medium': row['medium']
                    }
                articles_list.append(data)
            result_dict[i] = articles_list
        return result_dict
            
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


def save_results(start_date, end_date, topic_model, keep_headline, keep_teaser, keep_body, process_topics, df):
    if not os.path.exists(f'./modeling_results/{start_date}_{end_date}_{keep_headline}_{keep_teaser}_{keep_body}'):
        os.mkdir(f'./modeling_results/{start_date}_{end_date}_{keep_headline}_{keep_teaser}_{keep_body}')
    topic_model.topic_model.save(f'./modeling_results/{start_date}_{end_date}_{keep_headline}_{keep_teaser}_{keep_body}/{start_date}_{end_date}-{keep_headline}|{keep_teaser}|{keep_body}.pkl')
    df.to_csv(f'./modeling_results/{start_date}_{end_date}_{keep_headline}_{keep_teaser}_{keep_body}/original_docs.csv')
    labels_and_summaries_df = pd.DataFrame(process_topics.topic_labels, columns=['topic', 'label'])
    labels_and_summaries_df['summary'] = [x[1] for x in process_topics.topic_summaries]
    labels_and_summaries_df.to_csv(f'./modeling_results/{start_date}_{end_date}_{keep_headline}_{keep_teaser}_{keep_body}/labels_and_summaries.csv')


def load_results(start_date, end_date, keep_headline, keep_teaser, keep_body):
    directory = f'./modeling_results/{start_date}_{end_date}_{keep_headline}_{keep_teaser}_{keep_body}'
    current_directory = os.getcwd()
    try:
        labels_and_summaries_df = pd.read_csv(f'{directory}/labels_and_summaries.csv', index_col=0)
        original_docs = pd.read_csv(f'{directory}/original_docs.csv', index_col=0)
        os.chdir(directory)
        topic_model = BERTopic.load(f'./{start_date}_{end_date}-{keep_headline}|{keep_teaser}|{keep_body}.pkl')
        os.chdir(current_directory)
        return original_docs, labels_and_summaries_df, topic_model
    except:
        raise Exception('no model for this time frame')
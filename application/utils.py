import os, uuid, shutil, datetime, json
from sqlalchemy import create_engine, MetaData, Table, select, inspect, and_, or_
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

from google.cloud import aiplatform, storage
from vertexai.language_models import TextGenerationModel


class LoadArticles():
    def __init__(self):
        self.engine = create_engine(os.environ.get('SQLALCHEMY_DATABASE_URI'))
        self.metadata = MetaData()
        self.tables = inspect(self.engine).get_table_names()

    def load_articles(self, start_date, end_date, include_paywalled):
        articles = []
        for table in self.tables:
            table_data = Table(table, self.metadata, autoload_with=self.engine)
            condition1 = table_data.c.date_published >= start_date
            condition2 = table_data.c.date_published <= end_date
            if include_paywalled == True:
                condition3 = or_(table_data.c.paywall == True, table_data.c.paywall == False)
            else:
                condition3 = table_data.c.paywall == False
            query = select(table_data).where(and_(condition1, condition2, condition3))
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
    def __init__(self, kicker, headline, teaser, body, medium_list):
        self.kicker = kicker
        self.headline = headline
        self.teaser = teaser
        self.body = body
        self.medium_list = medium_list

    def filter_articles(self, df):
        common_indices = []
        if self.kicker == True:
            filtered_df = df[(df['kicker'].notna()) & (df['kicker'] != '')]
            kicker_indices = list(filtered_df.index)
            common_indices.append(set(kicker_indices))
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
            result_df['full_content'] = result_df['kicker'] + ' ' + result_df['headline'] + ' ' + result_df['teaser'] + ' ' + result_df['body']

        return result_df[result_df['medium'].isin(self.medium_list)]


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

    def transform(self, baseline_model_path, docs):
        base_model = BERTopic.load(baseline_model_path)
        topics, probabilities = base_model.transform(docs)
        return topics, probabilities


class ProcessTopics():
    def __init__(self, topic_model, docs, docs_df, google_project_id):
        self.topic_model = topic_model
        self.docs = docs
        self.docs_df = docs_df
        self.top_topics = pd.DataFrame([{'topic': row['Topic'], 'representation': row['Representation'], 'num_docs': row['Count']} for i, row in self.topic_model.get_topic_info().iloc[1:11].iterrows()])
        self.documents_df = topic_model.get_document_info(docs, df=docs_df)
        self.num_articles_per_medium = {}
        self.google_project_id = google_project_id
        aiplatform.init(project=google_project_id)
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
                labels.append((i, topic_label.text))
                print(i, topic_label)
                # if topic_label.text != '':
                #     labels.append((i, topic_label.text))
                # else:
                #     labels.append((i, self.top_topics[i]['representation']))
            except Exception as e:
                print(f'Exception from compute_llm_topic_labels(): {e}')
                labels.append((i, 'foo'))
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
                summaries.append((i, topic_summary.text))
                print(i, topic_summary)
                # if topic_summary.text != '':
                #     summaries.append((i, topic_summary.text))
                # else:
                #     summaries.append((i, self.top_topics[i]['representation']))
            except Exception as e:
                print(f'Exception from compute_llm_topic_summaries(): {e}')
                summaries.append((i, 'foo'))
        self.topic_summaries = summaries

    def build_article_data(self):
        result_dict = {}
        for i in range(10):
            articles_list = []
            articles = self.documents_df[self.documents_df['Topic'] == i].sort_values(by='Probability', ascending=False)
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


def save_results(start_date, end_date, topic_model, keep_kicker, keep_headline, keep_teaser, keep_body, process_topics):
    unique_key = str(uuid.uuid4())
    client = storage.Client(project='protean-unity-398412')
    bucket = client.get_bucket('balancr-models-bucket')
    topic_model.topic_model.save(f'./modeling_results/{unique_key}_model', serialization='safetensors', save_ctfidf=False)
    blob = bucket.blob(f'modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/model/config.json')
    blob.upload_from_filename(f'./modeling_results/{unique_key}_model/config.json')
    blob = bucket.blob(f'modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/model/topics.json')
    blob.upload_from_filename(f'./modeling_results/{unique_key}_model/topics.json')
    blob = bucket.blob(f'modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/model/topic_embeddings.safetensors')
    blob.upload_from_filename(f'./modeling_results/{unique_key}_model/topic_embeddings.safetensors')
    process_topics.documents_df.to_parquet(f'gs://balancr-models-bucket/modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/docs.parquet')
    labels_and_summaries_df = pd.DataFrame(process_topics.topic_labels, columns=['topic', 'label'])
    labels_and_summaries_df['summary'] = [x[1] for x in process_topics.topic_summaries]
    labels_and_summaries_df.to_parquet(f'gs://balancr-models-bucket/modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/labels_and_summaries.parquet')
    shutil.rmtree(f'./modeling_results/{unique_key}_model')


def load_results(start_date, end_date, keep_kicker, keep_headline, keep_teaser, keep_body):
    unique_key = str(uuid.uuid4())
    os.mkdir(f'./modeling_results/{unique_key}_model_tmp')
    try:
        client = storage.Client(project='protean-unity-398412')
        bucket = client.get_bucket('balancr-models-bucket')
        blob = bucket.blob(f'modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/model/config.json')
        blob.download_to_filename(f'./modeling_results/{unique_key}_model_tmp/config.json')
        blob = bucket.blob(f'modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/model/topics.json')
        blob.download_to_filename(f'./modeling_results/{unique_key}_model_tmp/topics.json')
        blob = bucket.blob(f'modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/model/topic_embeddings.safetensors')
        blob.download_to_filename(f'./modeling_results/{unique_key}_model_tmp/topic_embeddings.safetensors')
        model = BERTopic.load(f'./modeling_results/{unique_key}_model_tmp')
        shutil.rmtree(f'./modeling_results/{unique_key}_model_tmp')
        docs = pd.read_parquet(f'gs://balancr-models-bucket/modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/docs.parquet')
        labels_and_summaries_df = pd.read_parquet(f'gs://balancr-models-bucket/modeling_results/{start_date}_{end_date}_{keep_kicker}|{keep_headline}|{keep_teaser}|{keep_body}/labels_and_summaries.parquet')
        return docs, labels_and_summaries_df, model
    except Exception as e:
        shutil.rmtree(f'./modeling_results/{unique_key}_model_tmp')
        raise Exception('no model for this time frame')


def format_prediction_data(topics, probabilities, df, metadata):
    articles = []
    for i, v in enumerate(topics):
        prediction = {
            'article_id': int(df.iloc[i]['id']),
            'source': df.iloc[i]['medium'],
            'topic': int(v),
            'probability': float(probabilities[i])
        }
        articles.append(prediction)
    predictions = {
        'articles': articles,
        'metadata': metadata
    }
    return predictions


def save_predictions(topics, probabilities, filtered_df, baseline_model, start_date, end_date, keep_kicker, keep_headline, keep_teaser, keep_body, include_paywalled, sources):
    filename = str(uuid.uuid4())
    client = storage.Client(project='protean-unity-398412')
    bucket = client.get_bucket('balancr-models-bucket')
    blobs = client.list_blobs('balancr-models-bucket', prefix='modeling_results/predictions')
    sources.sort()
    metadata = {
        'baseline_model': baseline_model,
        'start_date': datetime.datetime.strftime(start_date, '%Y-%m-%d'),
        'end_date': datetime.datetime.strftime(end_date, '%Y-%m-%d'),
        'sources': ','.join(sources),
        'keep_kicker': str(keep_kicker),
        'keep_headline': str(keep_headline),
        'keep_teaser': str(keep_teaser),
        'keep_body': str(keep_body),
        'include_paywalled': str(include_paywalled)
    }
    for blob in blobs:
        if metadata == blob.metadata:
            raise Exception('Predictions for these articles already exist')
        else: 
            predictions = format_prediction_data(topics, probabilities, filtered_df, metadata)
            json_predictions = json.dumps(predictions)
            blob = bucket.blob(f'modeling_results/predictions/{filename}.json')
            blob.metadata = metadata
            blob.upload_from_string(json_predictions, content_type="application/json")
            return predictions


def load_predictions(baseline_model, start_date, end_date, keep_kicker, keep_headline, keep_teaser, keep_body, include_paywalled, sources):
    client = storage.Client(project='protean-unity-398412')
    blobs = client.list_blobs('balancr-models-bucket', prefix='modeling_results/predictions')
    sources.sort()
    metadata = {
        'baseline_model': baseline_model,
        'start_date': datetime.datetime.strftime(start_date, '%Y-%m-%d'),
        'end_date': datetime.datetime.strftime(end_date, '%Y-%m-%d'),
        'sources': ','.join(sources),
        'keep_kicker': str(keep_kicker),
        'keep_headline': str(keep_headline),
        'keep_teaser': str(keep_teaser),
        'keep_body': str(keep_body),
        'include_paywalled': str(include_paywalled)
    }
    for blob in blobs:
        if metadata == blob.metadata:
            result = json.loads(blob.download_as_text())
            return result
    raise Exception('no predictions with these parameters')



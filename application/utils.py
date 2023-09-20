import os
from sqlalchemy import create_engine, MetaData, Table, select, inspect, and_


class LoadArticles():
    def __init__(self):
        # self.engine = create_engine(f'postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}?sslmode=require')
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
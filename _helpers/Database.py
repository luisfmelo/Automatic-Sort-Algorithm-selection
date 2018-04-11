import pandas as pd
from sqlalchemy import create_engine

from config import DB_PATH


class Database:
    def __init__(self):
        self.db = create_engine(DB_PATH)

    def get_database(self):
        return self.db

    def get_df_from_table(self, table_name):
        return pd.read_sql_table(table_name=table_name, con=self.db)

    def append_df_to_table(self, df, table_name, index=id):
        pd.DataFrame(df).to_sql(table_name, self.db, if_exists='append', index_label=id)

    def save_csv(self, table_name, path):
        self.get_df_from_table(table_name).to_csv(path)

    def load_csv(self, path):
        return pd.read_csv(path)

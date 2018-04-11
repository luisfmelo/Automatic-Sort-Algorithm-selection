import os

DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'
TRAIN_DATA = 'csv_files/train-arrays.csv'  # 'train-arrays.csv'
TARGET_OUTPUT = 'csv_files/train-target.csv'
DB_FEATURE_TABLE = 'train_features'
DB_PATH = 'sqlite:///{}/database.db'.format(DIR)

import os

DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'
TRAIN_DATA = 'csv_files/train-arrays.csv'  # 'train-arrays.csv'
TRAIN_TARGET_OUTPUT = 'csv_files/train-target.csv'
TEST_DATA = 'csv_files/test-arrays.csv'
DB_TRAIN_FEATURE_TABLE = 'train_features_data'
DB_TEST_FEATURE_TABLE = 'test_features_data'
DB_OUTPUT_TABLE = 'output_features'
DB_PATH = 'sqlite:///{}/database.db'.format(DIR)

SLACK_TOKEN = '8acd0498fb41a16e6b1b202ff4cd48e1'

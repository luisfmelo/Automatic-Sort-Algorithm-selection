import os

DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'   # Automatic-Sort-Algorithm-selection/'
# DIR_ = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/'
TRAIN_DATA = 'csv_files/train-arrays.csv'  # 'train-arrays.csv'
TRAIN_TARGET_OUTPUT = 'csv_files/train-target.csv'
TEST_DATA = 'csv_files/test-arrays.csv'
DB_TRAIN_FEATURE_TABLE = 'train_features_data'
DB_TEST_FEATURE_TABLE = 'test_features_data'
DB_OUTPUT_TABLE = 'output_features'
DB_PATH = 'sqlite:///{}/database.db'.format(DIR)

SLACK_TOKEN = 'xoxb-359976621911-4H0Z3pab9hsjqKzJVdFz9Kx0'

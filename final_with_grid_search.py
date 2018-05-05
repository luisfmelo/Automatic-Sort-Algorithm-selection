from datetime import datetime

from _helpers.GenericModel import GenericModel
from _helpers.SlackHelper import Slack
from config import DB_TEST_FEATURE_TABLE, DIR, TEST_DATA, TRAIN_TARGET_OUTPUT, TRAIN_DATA, DB_TRAIN_FEATURE_TABLE

CHUNKSIZE = 10
slack = Slack()
slack.send('Starting...')

# FEATURE ENGINEERING
start_time = datetime.now()
iterations = GenericModel.extract_features(DIR + TRAIN_TARGET_OUTPUT, DIR + TRAIN_DATA, CHUNKSIZE, DB_TRAIN_FEATURE_TABLE, FORCE=False)
if iterations == 0:
    print('Extract Feature is disabled. CSV file was generated')
else:
    delta_time = (datetime.now() - start_time)
    print('Feature Extraction ended in {} seconds: completed {} iterations of {} chunk'.format(delta_time, iterations, CHUNKSIZE))

GenericModel.extract_features(None, DIR + TEST_DATA, CHUNKSIZE, DB_TEST_FEATURE_TABLE, FORCE=False)

# Grid Search for every model for various combinations of features
arr_feature_cols = [
    'length',
    # 'max',
    # 'min',
    'dist_min_max',
    'average',
    'mean',
    # 'q1', 'q2', 'q3',
    'std_deviation',
    'variance',
    'avg_diff',
    'sorted_percentage'
]


algorithms = ['svm', 'random_forest', 'multinomial_naive_bayes', 'gaussian_naive_bayes', 'knn', 'decision_tree']

for algorithm in algorithms:
    from itertools import combinations

    for length in range(1, 6):
        comb = combinations([1, 2, 3, 4, 5, 6], length)

        for feature_cols in list(comb):
            GenericModel.grid_search(algorithm, feature_cols, 0.3, slack)

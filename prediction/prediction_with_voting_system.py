from datetime import datetime

from _helpers.GenericModel import GenericModel
from config import DB_TEST_FEATURE_TABLE, DIR, TEST_DATA, TRAIN_TARGET_OUTPUT, TRAIN_DATA, DB_TRAIN_FEATURE_TABLE

CHUNKSIZE = 10
test_size = 0.3

#FEATURE ENGINEERING
start_time = datetime.now()

# if EXTRACT_FEATURES:
iterations = GenericModel.extract_features(DIR + TRAIN_TARGET_OUTPUT, DIR + TRAIN_DATA, CHUNKSIZE, DB_TRAIN_FEATURE_TABLE, FORCE=False)
if iterations == 0:
    print('Extract Feature is disabled. CSV file was generated')
else:
    delta_time = (datetime.now() - start_time)
    print('Feature Extraction ended in {} seconds: completed {} iterations of {} chunk'.format(delta_time, iterations, CHUNKSIZE))

GenericModel.extract_features(None, DIR + TEST_DATA, CHUNKSIZE, DB_TEST_FEATURE_TABLE, FORCE=False)
###########################################################################

bin_models = []
feature_cols = ['length']  #, 'sorted_percentage', 'dist_min_max']

# RANDOM FOREST
algorithm = 'random_forest'
bin_model_file_name = 'random_forest.sav'
parameters = {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 500}
bin_models.append(GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size))

# DECISION TREE
# algorithm = 'decision_tree'
# bin_model_file_name = 'decision_tree_with_length.sav'
# parameters = {'criterion': 'entropy', 'random_state': 0}
# bin_models.append(GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size))

# SVM
algorithm = 'svm'
bin_model_file_name = 'svm.sav'
parameters = {}  # {"gamma": 0.001}
bin_models.append(GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size))

# Neural Networks
# algorithm = 'neural_networks'
# bin_model_file_name = 'neural_networks.sav'
# parameters = {
#     # "solver": "lbfgs",
#     # "alpha": "0.00005",
#     "hidden_layer_sizes": (5, 2),
#     # "random_state": 0
# }
# bin_models.append(GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size))

# Gradient Boosting Classifier
algorithm = 'gradient_boost'
bin_model_file_name = 'gradient_boost.sav'
parameters = {"n_estimators": 50, "learning_rate": 0.05}
bin_models.append(GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size))

GenericModel.predict_with_voting_system(bin_models, feature_cols, 'new_voting_system.csv')

# feature_cols = [
#     'length',
#     'max',
#     'min',
#     'dist_min_max',
#     # 'average',
#     # 'mean',
#     # 'q1', 'q2', 'q3',
#     # 'std_deviation',
#     # 'variance',
#     'avg_diff',
#     'sorted_percentage',
#     # 'target'
# ]

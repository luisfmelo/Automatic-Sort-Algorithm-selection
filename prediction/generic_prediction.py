from datetime import datetime

from _helpers.GenericModel import GenericModel
from config import DB_TEST_FEATURE_TABLE, DIR, TEST_DATA, TRAIN_TARGET_OUTPUT, TRAIN_DATA, DB_TRAIN_FEATURE_TABLE

CHUNKSIZE = 10

####### FEATURE ENGINEERING
start_time = datetime.now()

# if EXTRACT_FEATURES:
iterations = GenericModel.extract_features(DIR + TRAIN_TARGET_OUTPUT, DIR + TRAIN_DATA, CHUNKSIZE, DB_TRAIN_FEATURE_TABLE, FORCE=False)
if iterations == 0:
    print('Extract Feature is disabled. CSV file was generated')
else:
    delta_time = (datetime.now() - start_time)
    print('Feature Extraction ended in {} seconds: completed {} iterations of {} chunk'.format(delta_time, iterations, CHUNKSIZE))
###########################################################################

GenericModel.extract_features(None, DIR + TEST_DATA, CHUNKSIZE, DB_TEST_FEATURE_TABLE, FORCE=False)

feature_cols = ['length']

# any
# algorithm = 'any'
# bin_model_file_name = 'any.sav'
# output_file_name = 'any.csv'
# test_size = 0.1
#
# bin_model = GenericModel.apply_model(algorithm, {}, feature_cols, bin_model_file_name, test_size)
# GenericModel.predict(bin_model, feature_cols, output_file_name)



# # DECISION TREE
# algorithm = 'decision_tree'
# bin_model_file_name = 'decision_tree_with_length.sav'
# output_file_name = 'decision_tree.csv'
# test_size = 0.1
#
# parameters = {
#     'criterion': 'entropy',  # gini
#     'random_state': 0
# }
#
# bin_model = GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size)
# GenericModel.predict(bin_model, feature_cols, output_file_name)
#
# # KNN
# algorithm = 'knn'
# bin_model_file_name = 'knn.sav'
# output_file_name = 'knn_output.csv'
# test_size = 0.15
#
# parameters = {
#     'n_neighbors': 3,
#     'algorithm': 'ball_tree'
# }
#
# bin_model = GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size)
# GenericModel.predict(bin_model, feature_cols, output_file_name)
#
# # Gaussian Naive Bayes Classifier
# algorithm = 'gaussian_naive_bayes'
# bin_model_file_name = 'gaussian_naive_bayes.sav'
# output_file_name = 'gaussian_naive_bayes_output.csv'
# test_size = 0.15
#
# parameters = {}
#
# bin_model = GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size)
# GenericModel.predict(bin_model, feature_cols, output_file_name)
#
# # Multinomial Naive Bayes Classifier
# algorithm = 'multinomial_naive_bayes'
# bin_model_file_name = 'multinomial_naive_bayes.sav'
# output_file_name = 'multinomial_naive_bayes_output.csv'
# test_size = 0.15
#
# parameters = {}
#
# bin_model = GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size)
# GenericModel.predict(bin_model, feature_cols, output_file_name)
#
# # SVM
# algorithm = 'svm'
# bin_model_file_name = 'svm.sav'
# output_file_name = 'svm_output.csv'
# test_size = 0.15
#
# parameters = {}
#
# bin_model = GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size)
# GenericModel.predict(bin_model, feature_cols, output_file_name)

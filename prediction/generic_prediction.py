from _helpers.GenericModel import GenericModel

# DECISION TREE
feature_cols = ['length']
algorithm = 'decision_tree'
bin_model_file_name = 'decision_tree_with_length.sav'
output_file_name = 'new_output.csv'
test_size = 0.10

parameters = {
    'criterion': 'entropy',
    'random_state': 0
}

bin_model = GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size)

GenericModel.predict(bin_model, feature_cols, output_file_name)


# KNN
feature_cols = ['variance']
algorithm = 'knn'
bin_model_file_name = 'knn.sav'
output_file_name = 'knn_output.csv'
test_size = 0.15

parameters = {
    'n_neighbors': 2
}

bin_model = GenericModel.apply_model(algorithm, parameters, feature_cols, bin_model_file_name, test_size)

GenericModel.predict(bin_model, feature_cols, output_file_name)
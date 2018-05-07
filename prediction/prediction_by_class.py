from datetime import datetime

from _helpers.GenericModel import GenericModel
from config import DB_TEST_FEATURE_TABLE, DIR, TEST_DATA, TRAIN_TARGET_OUTPUT, TRAIN_DATA, DB_TRAIN_FEATURE_TABLE

CHUNKSIZE = 10
test_size = 0.3

# FEATURE ENGINEERING
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
classes = [
    {
        'len': 10,
        'algorithms': [
            {
                'name': 'random_forest',
                'parameters': {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 500},
                'feature_cols': ['sorted_percentage', 'dist_min_max']
            },
            {
                'name': 'decision_tree',
                'parameters': {'criterion': 'gini'},
                'feature_cols': ['dist_min_max']
            },
            {
                'name': 'neural_networks',
                'parameters': {"solver": 'lbfgs', "alpha": 1e-5, "hidden_layer_sizes": (5, 2), "random_state": 1},
                'feature_cols': ['sorted_percentage']
            }
        ]
    },

    {
        'len': 100,
        'algorithms': [
            {
                'name': 'random_forest',
                'parameters': {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 500},
                'feature_cols': ['sorted_percentage', 'dist_min_max']
            },
            {
                'name': 'decision_tree',
                'parameters': {'criterion': 'gini'},
                'feature_cols': ['dist_min_max']
            },
            {
                'name': 'neural_networks',
                'parameters': {"solver": 'lbfgs', "alpha": 1e-5, "hidden_layer_sizes": (5, 2), "random_state": 1},
                'feature_cols': ['sorted_percentage']
            }
        ]
    },

    {
        'len': 1000,
        'algorithms': [
            {
                'name': 'random_forest',
                'parameters': {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 500},
                'feature_cols': ['sorted_percentage', 'dist_min_max']
            },
            {
                'name': 'decision_tree',
                'parameters': {'criterion': 'gini'},
                'feature_cols': ['dist_min_max']
            },
            {
                'name': 'neural_networks',
                'parameters': {"solver": 'lbfgs', "alpha": 1e-5, "hidden_layer_sizes": (5, 2), "random_state": 1},
                'feature_cols': ['sorted_percentage']
            }
        ]
    },

    {
        'len': 10000,
        'algorithms': [
            {
                'name': 'random_forest',
                'parameters': {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 500},
                'feature_cols': ['sorted_percentage', 'dist_min_max']
            },
            {
                'name': 'decision_tree',
                'parameters': {'criterion': 'gini'},
                'feature_cols': ['dist_min_max']
            },
            {
                'name': 'neural_networks',
                'parameters': {"solver": 'lbfgs', "alpha": 1e-5, "hidden_layer_sizes": (5, 2), "random_state": 1},
                'feature_cols': ['sorted_percentage']
            }
        ]
    },

    {
        'len': 100000,
        'algorithms': [
            {
                'name': 'random_forest',
                'parameters': {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 500},
                'feature_cols': ['sorted_percentage', 'dist_min_max']
            },
            {
                'name': 'decision_tree',
                'parameters': {'criterion': 'gini'},
                'feature_cols': ['dist_min_max']
            },
            {
                'name': 'neural_networks',
                'parameters': {"solver": 'lbfgs', "alpha": 1e-5, "hidden_layer_sizes": (5, 2), "random_state": 1},
                'feature_cols': ['sorted_percentage']
            }
        ]
    },

    {
        'len': 1000000,
        'algorithms': [
            {
                'name': 'random_forest',
                'parameters': {'criterion': 'gini', 'max_depth': 6, 'max_features': 'log2', 'n_estimators': 500},
                'feature_cols': ['sorted_percentage', 'dist_min_max']
            },
            {
                'name': 'decision_tree',
                'parameters': {'criterion': 'gini'},
                'feature_cols': ['dist_min_max']
            },
            {
                'name': 'neural_networks',
                'parameters': {"solver": 'lbfgs', "alpha": 1e-5, "hidden_layer_sizes": (5, 2), "random_state": 1},
                'feature_cols': ['sorted_percentage']
            }
        ]
    }
]

# Train models
bin_models = []
for _class in classes:
    bin_model = {'len': _class['len'], 'models': []}

    # for each algorithm -> get bin file
    for algorithm in _class['algorithms']:
        bin_model['models'].append({
            'model_file': GenericModel.apply_model_by_class(algorithm, _class['len'], test_size),
            'feature_cols': algorithm['feature_cols']
        })

    bin_models.append(bin_model)


GenericModel.predict_with_voting_system_by_class(bin_models, 'voting_system.csv')

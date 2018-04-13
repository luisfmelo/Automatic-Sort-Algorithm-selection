import pickle

import numpy as np
import pandas as pd
import pymrmr as pymrmr
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from _helpers.Database import Database
from config import DIR, TEST_DATA, DB_PATH


class GenericModel:
    DB = Database(DB_PATH)

    @staticmethod
    def reformat_array():
        """
        input: string like '[123 324 567]'
            * transform to a list
            * cast elements to integer
        :return:
        """
        return lambda x: [int(elem) for elem in x[1:-1].split()]

    @staticmethod
    def extract_features(path_target_data, path_train_data, chunksize, table_name, FORCE=False):

        # If table already exists and is not to force, we only save that info to a csv file
        if GenericModel.DB.table_exists(table_name) and not FORCE:
            GenericModel.DB.db_table_to_csv(table_name, '../csv_files/{}.csv'.format(table_name))
            return 0

        iterations = 0
        feature_arr = []

        if path_target_data is not None:
            df_target = pd.read_csv(path_target_data, encoding='latin1', error_bad_lines=False, index_col='Id')

        for df_train in pd.read_csv(path_train_data, iterator=True, encoding='latin1', error_bad_lines=False, names=["id", "length", "array"],
                                    chunksize=chunksize):
            df_train['array'] = df_train['array'].apply(GenericModel.reformat_array())

            features = []
            for index, row in df_train.iterrows():
                array = np.array(row['array'])
                dist_between_elems = [array[i + 1] - array[i] for i in range(len(array) - 1)]
                # TODO - validations
                target = None if path_target_data is None else df_target['Predicted']

                features.append({
                    'id': row['id'],
                    'length': len(array),
                    'max': max(array),
                    'min': min(array),
                    'dist_min_max': max(array) - min(array),
                    # Compute the weighted average along the specified axis.
                    'average': np.average(array, axis=0),
                    # Compute the arithmetic mean along the specified axis
                    'mean': np.mean(array, axis=0),  # reduce(lambda x, y: x + y, array) / len(array),
                    'q1': np.percentile(array, 25),
                    'q2': np.percentile(array, 50),  # median
                    'q3': np.percentile(array, 75),
                    'std_deviation': np.std(array, axis=0),
                    'variance': np.var(array, axis=0),
                    'avg_diff': sum(dist_between_elems) / len(dist_between_elems),
                    'target': int(target[row['id']]) if target is not None else '',
                })

            GenericModel.DB.append_df_to_table(features, table_name, id)
            feature_arr = feature_arr + features
            iterations = iterations + 1

        return iterations

    @staticmethod
    def apply_model(algorithm, parameters, feature_cols, model_file, test_size=0.15):
        try:
            algorithm = GenericModel.get_algorithm(algorithm, parameters)
            classifier = algorithm['function']
            classifier_name = algorithm['name']
            classifier_parameters = algorithm['parameters']
        except ModuleNotFoundError as e:
            return str(e)

        dataset = GenericModel.DB.load_csv('../csv_files/features.csv')

        feature_arr = [dataset.columns.get_loc(feature_name) for feature_name in feature_cols]
        X = dataset.iloc[:, feature_arr].values

        y = dataset.iloc[:, dataset.columns.get_loc("target")].values

        # Get Train and Test Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)

        # df_pymrmr = dataset[-1:] + dataset[:-1]
        # a = pymrmr.mRMR(df_pymrmr, 'MIQ', 5)

        # Fit to model
        classifier.fit(X_train, y_train)

        # Predict Output
        y_pred = classifier.predict(X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # Get Accuracy
        accuracy = accuracy_score(y_test, y_pred) * 100

        print('--------------------------------------------')
        print(classifier_name)
        print('Parameters: ' + classifier_parameters)
        print("Accuracy with Decision Tree is: {} %".format(accuracy))
        print('Confusion Matrix: {}'.format(cm))

        # PUT EVERYTHING TO TRAIN
        classifier.fit(X, y)

        # save the model to disk
        filename = '../bin_models/' + model_file
        pickle.dump(classifier, open(filename, 'wb'))

        return filename

    @staticmethod
    def predict(bin_model, feature_cols, output_file_name):
        # Extract Features
        chunksize = 10
        GenericModel.extract_features(None, DIR + TEST_DATA, chunksize, 'test_feature_data')
        df_features = GenericModel.DB.get_df_from_table('test_feature_data')

        # load saved model from disk
        feature_arr = [df_features.columns.get_loc(feature_name) for feature_name in feature_cols]
        X_test_data = df_features.iloc[:, feature_arr].values

        loaded_model = pickle.load(open(bin_model, 'rb'))
        result = loaded_model.predict(X_test_data)

        df_output = pd.DataFrame()
        df_output['Id'] = df_features["id"]
        df_output['Predicted'] = result

        # save to CSV
        df_output.to_csv(output_file_name, index=False, columns=['Id', 'Predicted'], header=True)

    @staticmethod
    def get_algorithm(algorithm_code, parameters):
        if algorithm_code == 'decision_tree':
            algorithm = {
                'name': 'Decision Tree Classifier',
                'function': DecisionTreeClassifier(**parameters),
            }

        elif algorithm_code == 'knn':
            algorithm = {
                'name': 'K Neighbors Classifier',
                'function': KNeighborsClassifier(**parameters),
            }

        else:
            raise ModuleNotFoundError('Algorithm not available.')

        # String with parameters used
        algorithm['parameters'] = '; '.join([parameter + ': ' + str(value) for parameter, value in parameters.items()])

        return algorithm

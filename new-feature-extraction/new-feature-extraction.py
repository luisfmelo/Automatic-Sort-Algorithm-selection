import os
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.random.mtrand import permutation
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sqlalchemy import create_engine

# TRAIN_DATA = '../feature-extraction-min/train-arrays-min.csv'  # 'train-arrays.csv'
TRAIN_DATA = '../csv_files/train-arrays.csv'  # 'train-arrays.csv'
TARGET_OUTPUT = '../csv_files/train-target.csv'
EXTRACT_FEATURES = False


def reformat_array():
    """
    input: string like '[123 324 567]'
        * transform to a list
        * cast elements to integer
    :return:
    """
    return lambda x: [int(elem) for elem in x[1:-1].split()]


def feature_extraction(db, chunksize):
    table_name = 'train_features'

    if EXTRACT_FEATURES:
        feature_extraction_start_time = datetime.now()
        iterations = 0
        feature_arr = []

        df_target = pd.read_csv(TARGET_OUTPUT, encoding='latin1', error_bad_lines=False, index_col='Id')

        for df_train in pd.read_csv(TRAIN_DATA, iterator=True, encoding='latin1', error_bad_lines=False, names=["id", "length", "array"],
                                    chunksize=chunksize):
            df_train['array'] = df_train['array'].apply(reformat_array())

            features = []
            for index, row in df_train.iterrows():
                array = np.array(row['array'])
                # TODO - validations
                target = df_target['Predicted']

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
                    'target': int(target[row['id']])
                })

            pd.DataFrame(features).to_sql(table_name, db, if_exists='append', index_label=id)
            feature_arr = feature_arr + features
            iterations = iterations + 1
        delta_time = (datetime.now() - feature_extraction_start_time)
        print('Feature Extraction ended in {} seconds: completed {} iterations of {} chunk'.format(delta_time, iterations, chunksize))
        df = pd.DataFrame(feature_arr)
    else:
        df = pd.read_sql_table(table_name=table_name, con=db)

    return df


def knn_model(k, train, test, feature_cols, predict_col):
    # Create the knn model.
    # Look at the five closest neighbors.
    knn = KNeighborsRegressor(n_neighbors=k)
    # Fit the model on the training data.
    knn.fit(train[feature_cols], train[predict_col])
    # Make point predictions on the test set using the fit model.
    predictions = knn.predict(test[feature_cols])
    # probs = knn.predict_proba(test[feature_cols])

    return predictions


def apply_model(model, df):
    train_cols = ['length', 'max', 'min', 'dist_min_max', 'average', 'mean', 'q1', 'q2', 'q3', 'std_deviation', 'variance', 'target']
    target_col = ['target']

    # Randomly shuffle the index of nba.
    random_indices = permutation(df.index)
    # Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
    test_cutoff = np.math.floor(len(df) / 3)
    # Generate the test set by taking the first 1/3 of the randomly shuffled indices.
    test = df.loc[random_indices[1:test_cutoff]]
    # Generate the train set with the rest of the data.
    train = df.loc[random_indices[test_cutoff:]]

    if model == 'knn':
        k = 2
        predictions = knn_model(k, train, test, train_cols, target_col)
    else:
        raise Exception('No available model was selected.')

    # Get the actual values for the test set.
    actual = test[target_col]

    # Compute the mean squared error of our predictions.
    error = (((predictions - actual) ** 2).sum()) / len(predictions)

    return error['target']


def main():
    chunksize = 10
    db_name = 'database.db'
    start_time = datetime.now()
    if EXTRACT_FEATURES:
        try:
            os.remove(db_name)
        except OSError:
            pass
    db = create_engine('sqlite:///' + db_name)

    df = feature_extraction(db, chunksize)
    error = apply_model('knn', df)

    delta_time = (datetime.now() - start_time)
    print('Ended in {} seconds. Error: {}'.format(delta_time, error))


main()

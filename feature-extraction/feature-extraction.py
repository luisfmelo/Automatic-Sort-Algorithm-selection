import os
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.random.mtrand import permutation
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from _helpers.Database import Database
from config import TARGET_OUTPUT, TRAIN_DATA, DB_FEATURE_TABLE, DB_PATH

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

        db.append_df_to_table(features, DB_FEATURE_TABLE, id)
        feature_arr = feature_arr + features
        iterations = iterations + 1

    return iterations


def main():
    chunksize = 10
    start_time = datetime.now()
    if EXTRACT_FEATURES:
        try:
            os.remove(DB_PATH)
        except OSError:
            pass

        db = Database()
        iterations = feature_extraction(db, chunksize)
        delta_time = (datetime.now() - start_time)
        print('Feature Extraction ended in {} seconds: completed {} iterations of {} chunk'.format(delta_time, iterations, chunksize))
    else:
        print('Extract Feature is disabled.')


main()
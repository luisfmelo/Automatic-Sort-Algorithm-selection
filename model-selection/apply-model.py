from datetime import datetime

import pandas as pd
from numpy.random.mtrand import permutation
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from _helpers.Database import Database
from config import DB_FEATURE_TABLE


def knn_model(k, train, test, feature_cols, predict_col):
    # Create the knn model.
    # Look at the five closest neighbors.

    knn = KNeighborsRegressor(n_neighbors=k, )
    # Fit the model on the training data.
    knn.fit(train[feature_cols], train[predict_col])
    # Make point predictions on the test set using the fit model.
    predictions = knn.predict(test[feature_cols])
    # probs = knn.predict_proba(test[feature_cols])

    return predictions


def apply_model(model, df):
    train_cols = ['length']  # , 'max', 'min', 'dist_min_max', 'average', 'mean', 'q1', 'q2', 'q3', 'std_deviation', 'variance', 'target']
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
    start_time = datetime.now()

    db = Database()
    df = db.get_df_from_table(DB_FEATURE_TABLE)
    error = apply_model('knn', df)

    delta_time = (datetime.now() - start_time)
    print('Ended in {} seconds. Error: {}'.format(delta_time, error))


main()

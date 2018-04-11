import pandas as pd
import numpy as np

from _helpers.Database import Database
from config import DB_PATH


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
                if path_target_data is not None:
                    target = df_target['Predicted']
                else:
                    target = None

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

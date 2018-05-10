from itertools import combinations

from _helpers.GenericModel import GenericModel
from _helpers.LoggerHelper import Logger


def bulk(algorithm, length_class):
    results = []
    arr_feature_cols = ['average', 'mean', 'std_deviation', 'variance', 'avg_diff', 'sorted_percentage']

    for i in range(1, len(arr_feature_cols) + 1):
        comb = combinations(arr_feature_cols, i)

        for feature_cols in list(comb):
            try:
                parameters, acc = GenericModel.grid_search(algorithm, feature_cols, 0.3, class_len=length_class)
                results.append({'parameters': parameters, 'accuracy': eval(acc), 'features': feature_cols})
            except Exception as e:
                print(str(e))

    best_match = {'parameters': None, 'accuracy': 0, 'features': []}
    for result in results:
        if result['accuracy'] > best_match['accuracy']:
            best_match = result

    Logger.send_personal('************************************')
    Logger.send_personal('Algorithm: {}'.format(algorithm))

    if best_match['parameters'] is not None:
        Logger.send_personal('Best Match')
        Logger.send_personal('Accuracy: {} %'.format(str(best_match['accuracy'])))
        Logger.send_personal('Parameters: {}'.format(str(best_match['parameters'])))
        Logger.send_personal('Features: {}'.format('[{}]'.format(', '.join(best_match['features']))))
    else:
        Logger.send_personal('Not Found')
    Logger.send_personal('************************************')


algorithms = [
    'decision_tree', 'knn', 'gaussian_naive_bayes',
    'multinomial_naive_bayes', 'svm', 'random_forest',
    'neural_networks', 'extra_trees', 'ada_boost',
    'gradient_boost']
lengths = [100, 10000]

for l in lengths:
    for a in algorithms:
        bulk(a, l)

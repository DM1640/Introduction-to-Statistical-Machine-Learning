import csv
import math
import time
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class DecisionStumpClassifier:

    def fit(self, x_data, y_data, observation_weights):
        N = len(x_data)

        self.err = best_err = 1.0
        self.i = best_i = 0
        self.threshold = best_threshold = 0.0
        self.dir = best_dir = 1

        for i, features in enumerate(zip(*x_data)):
            sorted_features = []
            sorted_y_data = []
            sorted_observation_weights = []

            for feature, y, w in sorted(zip(features, y_data, observation_weights)):
                sorted_features.append(feature)
                sorted_y_data.append(y)
                sorted_observation_weights.append(w)

            for j, (x, next_x) in enumerate(zip(sorted_features, sorted_features[1:])):
                if x == next_x:
                    continue
                threshold = x + 0.5 * (next_x - x)

                # dir = -1 => direction changes from -1 to  1
                # dir =  1 => direction changes from  1 to -1
                for _dir in (-1, 1):
                    sorted_preds = [_dir] * (j + 1) + [-_dir] * (N - j - 1)
                    err = sum(w if pred != y else 0.0 for pred, y, w in zip(sorted_preds, sorted_y_data, sorted_observation_weights))
                    if err >= best_err:
                        continue
                    best_err = err
                    best_i = i
                    best_threshold = threshold
                    best_dir = _dir

        self.err = best_err
        self.i = best_i
        self.threshold = best_threshold
        self.dir = best_dir

    def predict(self, x_data):
        i = self.i
        threshold = self.threshold
        _dir = self.dir

        return [_dir if x[i] < threshold else -_dir for x in x_data]


def read_wdbc_dataset(data_file):
    y_data = []
    x_data = []

    with open(data_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            x_data.append(list(map(lambda a: float(a), row[2:])))
            y_data.append(-1 if row[1] == 'M' else 1)

    return x_data, y_data


def adaboost_run(x_training_data, y_training_data, x_testing_data, y_testing_data, n_iters, h_factory, calc_err=False):
    errs_training = []
    errs_testing = []

    N = len(x_training_data)

    hs = []
    h_weights = []

    w = 1 / N
    observation_weights = [w] * N

    for t in range(n_iters):

        # create a weak classifer to classify training data using weights
        h = h_factory()
        h.fit(x_training_data, y_training_data, observation_weights)
        hs.append(h)
        preds = h.predict(x_training_data)

        # compute hypothesis error of this iteration
        err = sum(w if pred != y else 0.0 for pred, y, w in zip(preds, y_training_data, observation_weights))
        observation_weights_total = sum(observation_weights)
        err /= observation_weights_total

        # compute hypothesis weight for this iteration
        h_weight = 0.5 * (math.log((1 - err) / err))
        h_weights.append(h_weight)

        # update observation weights
        observation_weights = [w * math.exp(-h_weight * y * pred) for pred, y, w in zip(preds, y_training_data, observation_weights)]
        observation_weights_total = sum(observation_weights)
        observation_weights = [w / observation_weights_total for w in observation_weights]

        if not calc_err:
            continue

        # compute error for training data
        err = adaboost_calc_err(hs, h_weights, x_training_data, y_training_data)
        errs_training.append(err)

        if x_testing_data is not None and y_testing_data is not None:
            # compute error for testing data
            err = adaboost_calc_err(hs, h_weights, x_testing_data, y_testing_data)
            errs_testing.append(err)

    return hs, h_weights, errs_training, errs_testing


def adaboost_calc_err(hs, h_weights, x_data, y_data):
    N = len(x_data)

    preds = adaboost_predict(hs, h_weights, x_data)

    return sum(1.0 if pred != y else 0.0 for pred, y in zip(preds, y_data)) / N


def adaboost_predict(hs, h_weights, x_data):
    N = len(x_data)

    preds = [0] * N
    for h_weight, h in zip(h_weights, hs):
        _preds = h.predict(x_data)
        preds = [p + h_weight * _p for p, _p in zip(preds, _preds)]
    preds = list(map(lambda x: -1 if x < 0 else 1, preds))

    return preds


def run(x_training_data, y_training_data, x_testing_data, y_testing_data, n_iters, calc_err):
    print('* Number of iterations : {}'.format(n_iters))
    print('* Calculate errors : {}'.format(calc_err))

    print('>>> Running AdaBoost w/ DecisionTreeClassifier(max_depth=1) ...')
    start_time = time.monotonic()

    # weak classifier : sklearn.tree.DecisionTreeClassifier w/ depth=1
    _, _, errs_training_0, errs_testing_0 = adaboost_run(
        x_training_data, y_training_data,
        x_testing_data, y_testing_data,
        n_iters, lambda : DecisionTreeClassifier(max_depth=1), calc_err)
    print('Elapsed seconds : {:.6f} '.format(time.monotonic() - start_time))
    if calc_err:
        if len(errs_training_0) != 0:
            print('Error (training) : {:.6f}'.format(errs_training_0[-1]))
            if len(errs_testing_0) != 0:
                print('Error (testing) : {:.6f}'.format(errs_testing_0[-1]))

    print('>>> Running AdaBoost w/ DecisionTreeClassifier(max_depth=2) ...')
    start_time = time.monotonic()
    
    # weak classifier : sklearn.tree.DecisionTreeClassifier w/ depth=2
    _, _, errs_training_1, errs_testing_1 = adaboost_run(
        x_training_data, y_training_data,
        x_testing_data, y_testing_data,
        n_iters, lambda : DecisionTreeClassifier(max_depth=2), calc_err)
    print('Elapsed seconds : {:.6f}'.format(time.monotonic() - start_time))
    if calc_err:
        if len(errs_training_1) != 0:
            print('Error (training) : {:.6f}'.format(errs_training_1[-1]))
            if len(errs_testing_1) != 0:
                print('Error (testing) : {:.6f}'.format(errs_testing_1[-1]))

    print('>>> Running AdaBoost w/ DecisionStumpClassifier ...')
    start_time = time.monotonic()
    
    # weak classifier : DecisionStumpClassifier
    _, _, errs_training_2, errs_testing_2 = adaboost_run(
        x_training_data, y_training_data,
        x_testing_data, y_testing_data,
        n_iters, DecisionStumpClassifier, calc_err)
    print('Elapsed seconds : {:.6f}'.format(time.monotonic() - start_time))
    if calc_err:
        if len(errs_training_2) != 0:
            print('Error (training) : {:.6f}'.format(errs_training_2[-1]))
            if len(errs_testing_2) != 0:
                print('Error (testing) : {:.6f}'.format(errs_testing_2[-1]))

    if calc_err:
        T = len(errs_training_0)
        if T != 0 and T == len(errs_training_1) and T == len(errs_training_2):
            xs = list(range(1, T + 1))

            _, ax = plt.subplots(1, 1)
            ax.plot(xs, errs_training_0, label='DecisionTreeClassifier(max_depth=1)')
            ax.plot(xs, errs_training_1, label='DecisionTreeClassifier(max_depth=2)')
            ax.plot(xs, errs_training_2, label='DecisionStumpClassifier')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(T // 5))
            ax.set_yticks([i * 0.01 for i in range(10 + 1)])
            ax.set_xlabel('iteration')
            ax.set_ylabel('error')
            ax.set_title('Training Dataset')
            ax.legend(loc='upper right')

            _, ax = plt.subplots(1, 1)
            ax.plot(xs, errs_testing_0, label='DecisionTreeClassifier(max_depth=1)')
            ax.plot(xs, errs_testing_1, label='DecisionTreeClassifier(max_depth=2)')
            ax.plot(xs, errs_testing_2, label='DecisionStumpClassifier')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(T // 5))
            ax.set_yticks([i * 0.01 for i in range(10 + 1)])
            ax.set_xlabel('iteration')
            ax.set_ylabel('error')
            ax.set_title('Testing Dataset')
            ax.legend(loc='upper right')

            plt.show()


if __name__ == '__main__':
    
    x_data, y_data = read_wdbc_dataset('wdbc_data.csv')

    # split dataset
    x_training_data = x_data[:300]
    y_training_data = y_data[:300]
    x_testing_data = x_data[300:]
    y_testing_data = y_data[300:]

    print('Number of training records : {}'.format(len(x_training_data)))
    print('Number of testing records : {}'.format(len(x_testing_data)))

    # ---------- do NOT calculate errors
    run(x_training_data, y_training_data, x_testing_data, y_testing_data, 100, False)
    run(x_training_data, y_training_data, x_testing_data, y_testing_data, 200, False)
    run(x_training_data, y_training_data, x_testing_data, y_testing_data, 500, False)

    # ---------- do calculate errors
    run(x_training_data, y_training_data, x_testing_data, y_testing_data, 100, True)
    run(x_training_data, y_training_data, x_testing_data, y_testing_data, 200, True)
    run(x_training_data, y_training_data, x_testing_data, y_testing_data, 500, True)



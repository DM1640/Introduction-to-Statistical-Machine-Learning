#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import pandas
import cvxpy
import cvxopt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# switch to print detailed solving steps to console
VERBOSE = False
# max iterations
MAXITERS = 1000000

# use C of value 10.0 (suggested range is 1 ~ 100)
C = 100.0


# Primal algorithm (train)
def svm_train_primal(X, y, C):
    n, p = X.shape

    w = cvxpy.Variable(p)   # w
    b = cvxpy.Variable()    # b
    xi = cvxpy.Variable(n)  # ξ

    # minimize: 1 / 2 * ||w||^2 + C / n * ∑(ξ)
    prob = 0.5 * cvxpy.sum_squares(w) + C / n * cvxpy.sum(xi)
    # construct optimization problem
    # subject to: y * (w_T * x + b) >= 1 - ξ ξ >= 0
    problem = cvxpy.Problem(cvxpy.Minimize(prob), [cvxpy.multiply(y, (X @ w + b)) >= 1.0 - xi, xi >= 0.0])

    # solve the quadratic problem
    _ = problem.solve(verbose = VERBOSE, max_iters = MAXITERS, solver = 'ECOS')
    print('solution status:', problem.status)

    # only return w and b when a optimal solution is found
    if problem.status == 'optimal':
        return w.value, b.value
    return None, None


# Primal algorithm (predict)
def svm_predict_primal(X, y, model):
    # extract w and b
    w, b = model
    # perform prediction
    y_pred = X @ w + b
    # replace predicted values with 1 and -1
    y_pred[y_pred >= 0.0] = 1
    y_pred[y_pred < 0.0] = -1
    # convert predicted values to integer type
    y_pred = y_pred.astype(int)
    # return (modified) predicted values and its accuracy
    return y_pred, accuracy_score(y, y_pred)


# Dual algorithm (train)
def svm_train_dual(X, y, C):
    n, _ = X.shape

    y = y.reshape(-1, 1).astype(float)

    P = cvxopt.matrix((y * X) @ (y * X).T)
    q = cvxopt.matrix(-numpy.ones((n, 1)))
    G = cvxopt.matrix(numpy.vstack((numpy.eye(n), -numpy.eye(n))))
    h = cvxopt.matrix(numpy.hstack((numpy.ones(n) * C, numpy.zeros(n))))
    A = cvxopt.matrix(y.reshape(1, -1))
    b = cvxopt.matrix(numpy.zeros(1))

    cvxopt.solvers.options['show_progress'] = VERBOSE
    cvxopt.solvers.options['maxiters'] = MAXITERS

    # minimize: 1 / 2 * x_T * P * x + q_T * x      # 1 / 2 * α_T * P * α - α
    # subject to: G * x + s = h && s >= 0          # 0 <= α <= C
    #             A * x = b                        # ∑(α · y) = 0

    # solve the quadratic problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    print('solution status:', solution['status'])

    # only return w and b when a optimal solution is found
    if solution['status'] == 'optimal':
        alpha = numpy.array(solution['x'])
        # calculate w
        w = ((y * alpha).T @ X).reshape(-1, 1)
        # get candidates of b corresponding to positive alphas
        index = (alpha > 0.0).flatten()
        b = (y[index] - numpy.dot(X[index], w)).flatten()
        # iterate all candidates of b to see which one leads to the best accuracy
        best_b_ = b[0]
        best_accuracy = 0.0
        for b_ in b:
            # perform prediction
            y_pred = X @ w + b_
            # replace predicted values with 1 and -1
            y_pred[y_pred >= 0.0] = 1
            y_pred[y_pred < 0.0] = -1
            # convert predicted values to of integer type
            y_pred = y_pred.astype(int)
            accuracy = accuracy_score(y, y_pred)
            # compare accuracys to select the best one
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_b_ = b_
        return w.flatten(), best_b_
    return None, None


# Dual algorithm (predict)
def svm_predict_dual(X, y, model):
    # extract w and b
    w, b = model
    # perform prediction
    y_pred = X @ w + b
    # replace predicted values with 1 and -1
    y_pred[y_pred >= 0.0] = 1
    y_pred[y_pred < 0.0] = -1
    # convert predicted values to of integer type
    y_pred = y_pred.astype(int)
    # return (modified) predicted values and its accuracy
    return y_pred, accuracy_score(y, y_pred)


# Read dataset
def read_data():
    # read train data from the csv file to split into 2 arrays
    train_data = pandas.read_csv("train.csv", header = None).values
    train_X = train_data[:, 1:]
    train_y = train_data[:, 0]
    # for y, convert 0 to -1
    train_y[train_y == 0] = -1

    # read test data from the csv file to split into 2 arrays
    test_data = pandas.read_csv("test.csv", header = None).values
    test_X = test_data[:, 1:]
    test_y = test_data[:, 0]
    # for y, convert 0 to -1
    test_y[test_y == 0] = -1

    return train_X, train_y, test_X, test_y


def main():
    train_X, train_y, test_X, test_y = read_data()
    n, _ = train_X.shape

    print('\n------ primal ------')
    w, b = svm_train_primal(train_X, train_y, C * n)
    print('w:', w)
    print('b:', b)
    y_pred, score = svm_predict_primal(train_X, train_y, (w, b))
    print('error on training set:', 1.0 - score)
    y_pred, score = svm_predict_primal(test_X, test_y, (w, b))
    print('error on testing set:', 1.0 - score)

    print('\n------ dual ------')
    w, b = svm_train_dual(train_X, train_y, C)
    print('w:', w)
    print('b:', b)
    y_pred, score = svm_predict_dual(train_X, train_y, (w, b))
    print('error on training set:', 1.0 - score)
    y_pred, score = svm_predict_dual(test_X, test_y, (w, b))
    print('error on testing set:', 1.0 - score)

    print('\n------ primal (by sklearn, 3rd-party) ------')
    clf = LinearSVC(C = C, dual = False, max_iter = MAXITERS)
    clf.fit(train_X, train_y)
    w, b = clf.coef_.flatten(), clf.intercept_[0]
    print('w:', w)
    print('b:', b)
    y_pred = clf.predict(train_X)
    score = accuracy_score(train_y, y_pred)
    print('error on training set:', 1.0 - score)
    y_pred = clf.predict(test_X)
    score = accuracy_score(test_y, y_pred)
    print('error on testing set:', 1.0 - score)

    print('\n------ dual (by sklearn, 3rd-party) ------')
    clf = LinearSVC(C = C, dual = True, max_iter = MAXITERS)
    clf.fit(train_X, train_y)
    w, b = clf.coef_.flatten(), clf.intercept_[0]
    print('w:', w)
    print('b:', b)
    y_pred = clf.predict(train_X)
    score = accuracy_score(train_y, y_pred)
    print('error on training set:', 1.0 - score)
    y_pred = clf.predict(test_X)
    score = accuracy_score(test_y, y_pred)
    print('error on testing set:', 1.0 - score)


if __name__ == '__main__':
    main()

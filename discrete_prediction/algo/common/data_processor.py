# Created by Hansi at 11/22/2020
import numpy as np
from pyentrp import entropy as ent

from discrete_prediction.algo.common.label_generator import label_instances_for_classification


def transform_to_series(data, args):
    rows, cols = data.shape
    series = []
    X = []
    Y = []
    x_for_label = []

    # covert each array into series of data
    for i in range(cols):
        variable_values = data[:, i]
        variable_series = ent.util_pattern_space(variable_values, lag=1, dim=args['train_test_series_length'])
        variableX = variable_series[:, :args['train_series_length']]
        series.append(variableX)
        if i == 0:
            x_for_label = variableX
            Y = variable_series[:, -1]
    series = np.array(series)

    # convert into an array of shape #instances * #variable * series length
    x, y, z = series.shape
    for n in range(y):
        temp_array = []
        for variable_count in range(x):
            temp_array.append(series[variable_count][n])
        temp_array = np.array(temp_array)
        X.append(temp_array)

    X = np.array(X)
    return X, Y, x_for_label


def transform_to_test_series(data, args):
    rows, cols = data.shape
    series = []
    X = []

    # covert each array into series of data
    for i in range(cols):
        variable_values = data[:, i]
        variableX = ent.util_pattern_space(variable_values, lag=1, dim=args['train_series_length'])
        series.append(variableX)

    series = np.array(series)

    # convert into an array of shape #instances * #variable * series length
    x, y, z = series.shape
    for n in range(y):
        temp_array = []
        for variable_count in range(x):
            temp_array.append(series[variable_count][n])
        temp_array = np.array(temp_array)
        X.append(temp_array)

    X = np.array(X)
    return X


def format_training_data(train, val, args):
    trainX, trainY, train_x_for_label = transform_to_series(train, args)
    valX, valY, val_x_for_label = transform_to_series(val, args)

    # label data
    train_labels = label_instances_for_classification(train_x_for_label, trainY, args)
    val_labels = label_instances_for_classification(val_x_for_label, valY, args)

    return trainX, train_labels, valX, val_labels


def format_testing_data(test, args):
    return transform_to_test_series(test, args)

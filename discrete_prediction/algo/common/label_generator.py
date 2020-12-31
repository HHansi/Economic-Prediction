# Created by Hansi at 11/17/2020
import numpy as np

from discrete_prediction.args import fall_label, static_label, growth_label


def get_label(valY, valX):
    if valY < valX:
        label = fall_label
    elif valY == valX:
        label = static_label
    else:
        label = growth_label
    return label


def label_instances_for_classification(X, Y, args):
    """
    Generate labels for time series classification

    :param X: list of float
        X values which need to be compared with Y value while generating the label
    :param Y: list of float
        Y values
    :param args: json
    :return: list of labels(int)
        Labels corresponding to Y values
    """
    labels = list()
    for i in range(len(X)):
        # compare with immediate value
        trainY_element = Y[i]
        trainX_element = X[i, args['train_series_length'] - 1]
        label = get_label(trainY_element, trainX_element)
        labels.append(label)
    return labels


def discretize_series(series):
    """
    Convert continuous value series into discrete values.
    Since label generation compare current value to previous value, resulting series will be in length of n-1 when n
    length series is inserted

    :param series: list of float
    :return: list of labels(int)

    """
    labels = list()
    val0 = series[0]
    for i in range(1, len(series)):
        val1 = series[i]
        label = get_label(val1, val0)
        labels.append(label)
        val0 = val1
    return labels


# compute label distribution in givel labels
def count_labels(labels):
    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    return unique_elements, counts_elements

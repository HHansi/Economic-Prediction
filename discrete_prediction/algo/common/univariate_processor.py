# Created by Hansi at 11/22/2020
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyentrp import entropy as ent

from discrete_prediction.args import random_seed
from discrete_prediction.algo.common.label_generator import get_label


def create_split(data_file_path, args):
    # load data
    df = pd.read_csv(data_file_path, delimiter=',')
    print("Head of data: ")
    print(df.head())

    data_series = np.array(df[args['column_name']])
    print(f"Length of series: {len(data_series)}")

    # split data
    train, val = train_test_split(data_series, test_size=args['test_size'], shuffle=False, random_state=random_seed)
    print(f"train split : {train.shape}, test split: {val.shape}")

    if args['normalise']:
        # normalise data
        scaler = StandardScaler()
        train = scaler.fit_transform(train.reshape(-1, 1))
        val = scaler.transform(val.reshape(-1, 1))

    return train, val


def format_data(train, val, args):
    train = ent.util_pattern_space(train, lag=1, dim=args['train_test_series_length'])
    trainX = train[:, :args['train_series_length']]
    trainY = train[:, -1]

    val = ent.util_pattern_space(val, lag=1, dim=args['train_test_series_length'])
    valX = val[:, :args['train_series_length']]
    valY = val[:, -1]

    print(f"train series count: {len(trainX)}, test series count: {len(valX)}")

    # label data
    train_targs = list()
    for i in range(len(trainX)):
        # compare with immediate value
        trainY_element = trainY[i]
        trainX_element = trainX[i, args['train_series_length'] - 1]
        label = get_label(trainY_element, trainX_element)
        train_targs.append(label)

    val_targs = list()
    for i in range(len(valX)):
        valY_element = valY[i]
        valX_element = valX[i, args['train_series_length'] - 1]
        label = get_label(valY_element, valX_element)
        val_targs.append(label)

    # compute label distribution in training data set
    unique_elements_train, counts_elements_train = np.unique(train_targs, return_counts=True)
    print(f"Training data set label distribution: \n {np.asarray((unique_elements_train, counts_elements_train))}")

    # compute label distribution in test data set
    unique_elements_test, counts_elements_test = np.unique(val_targs, return_counts=True)
    print(f"Testing data set label distribution: \n {np.asarray((unique_elements_test, counts_elements_test))}")

    return trainX, train_targs, valX, val_targs



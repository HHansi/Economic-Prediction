# Created by Hansi at 12/5/2020
import pandas as pd

from continuous_prediction.algo.common.util import closestNumber, create_future_quarter_labels


def create_split(series, args):
    """
    Create train and test split

    :param series: list
    :param args: json
    :return: list, list
        train series and test series
    """
    future_n = args['future_n']
    test_split = args['test_split']

    test_size = int(len(series) * test_split)
    train_size = len(series) - test_size

    # for evaluation test data size need to be divisible by number of future predictions
    if test_size % future_n != 0:
        closest_num = closestNumber(test_size, future_n)
        diff = closest_num - test_size
        train_size = train_size - diff

    train, test = series[0:train_size], series[train_size:len(series)]
    return train, test


def create_df(time_data, originals, predictions):
    diff = len(predictions) - len(originals)
    new_time = create_future_quarter_labels(time_data[len(time_data) - 1], diff)
    print(new_time)
    time_data.extend(new_time)
    time_df = pd.DataFrame(time_data)

    org_df = pd.DataFrame(originals)
    pred_df = pd.DataFrame(predictions)
    df = pd.concat([time_df, org_df, pred_df], ignore_index=True, axis=1)
    df.columns = ['Time', 'Original', 'Prediction']
    df.set_index('Time', inplace=True)
    return df

# Created by Hansi at 12/30/2020
import os

import pandas as pd

from args import BASE_PATH
from discrete_prediction.algo.common.evaluator import evaluate
from discrete_prediction.algo.common.label_generator import discretize_series


def eval(input_file, output_file=None):
    """
    Convert continuous series in columns 'Original' and 'Prediction' to discrete values and evaluate.
    :param input_file: str
        Path to a .csv file with columns 'Original' and 'Prediction'
    :param output_file: str, optional
        Path to a .csv file.
        If a file path is given, converted predictions will be saved.
    :return:
    """
    df = pd.read_csv(input_file, delimiter=',')
    originals = df['Original'].dropna()
    predictions = df['Prediction']
    label_predictions_all = discretize_series(predictions)

    label_originals = discretize_series(originals)
    # remove future predictions
    n = len(label_originals)
    label_predictions = label_predictions_all[:n]

    print(f'original: {label_originals}')
    print(f'prediction: {label_predictions}')

    results = evaluate(label_originals, label_predictions, average='weighted')
    print(f'weighted: {results}')

    results = evaluate(label_originals, label_predictions, average='macro')
    print(f'macro: {results}')

    n = len(df['Prediction']) - 1
    original_append = ['na' for i in range(n-len(label_originals))]
    label_originals = ['na'] + label_originals + original_append
    label_predictions = ['na'] + label_predictions_all

    df['Original_d'] = label_originals
    df['Prediction_d'] = label_predictions
    if output_file is not None:
        df.to_csv(output_file, sep=',', encoding='utf-8', index=False)


def generate_labels(file_path):
    df = pd.read_csv(file_path, delimiter=',')
    data = df['Data']
    data_labels = discretize_series(data)
    data_labels = ['na'] + data_labels
    df['Labels'] = data_labels
    df.to_csv(file_path, sep=',', encoding='utf-8', index=False)


if __name__ == "__main__":
    input_file_path = os.path.join(BASE_PATH, "results/ARIMA/manufacturing-4.csv")
    output_file_path = os.path.join(BASE_PATH, "results/ARIMA/manufacturing-4-discrete.csv")
    eval(input_file_path, output_file_path)

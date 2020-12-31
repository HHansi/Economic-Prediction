# Created by Hansi at 11/30/2020
import csv

import numpy as np
import pandas as pd

from discrete_prediction.algo.common.evaluator import evaluate
from discrete_prediction.algo.common.label_generator import count_labels
from discrete_prediction.algo.common.multivariate_processor import create_split, format_training_data
from discrete_prediction.algo.saxvsm import train_multivariate_model, predict_model
from discrete_prediction.args import data_file_path, multivariate_data_processing_args, f1_label, recall_label, \
    precision_label, result_file_path
from utils.data_preprocessor import normalise_train_data, normalise_data

# # load, process and format data
# train, val = create_split(data_file_path, args=multivariate_data_processing_args)

# load data
df = pd.read_csv(data_file_path, delimiter=',')
print("Head of data: ")
print(df.head())

# filter columns
filtered_df = df[multivariate_data_processing_args['column_names']]
filtered_data = filtered_df.to_numpy()

# split data
train, val = create_split(filtered_data, args=multivariate_data_processing_args)

# normalise data
if multivariate_data_processing_args['normalise']:
    train = normalise_train_data(train)
    val = normalise_data(val)

trainX, train_labels, valX, val_labels = format_training_data(train, val, args=multivariate_data_processing_args)

# calculate label distribution
unique_elements, counts_elements = count_labels(train_labels)
print(f"Training data set label distribution: \n {np.asarray((unique_elements, counts_elements))}")
unique_elements, counts_elements = count_labels(val_labels)
print(f"Testing data set label distribution: \n {np.asarray((unique_elements, counts_elements))}")

# open results file
csv_file = open(result_file_path, 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file, delimiter='\t')

window_size_max = multivariate_data_processing_args['train_series_length']
window_size_min = 2

for temp_window_size in range(window_size_min, window_size_max + 1):
    word_size_max = temp_window_size
    word_size_min = 2
    for temp_word_size in range(word_size_min, word_size_max + 1):
        n_bins_max = temp_word_size
        n_bins_min = 2
        for temp_n_bins in range(n_bins_min, n_bins_max + 1):
            temp_saxsvm_args = dict()
            temp_saxsvm_args['n_bins'] = temp_n_bins
            temp_saxsvm_args['word_size'] = temp_word_size
            temp_saxsvm_args['window_size'] = temp_window_size

            # train model
            model = train_multivariate_model(trainX, train_labels, args=temp_saxsvm_args)

            # validate model
            str_val_labels = ' '.join(str(x) for x in val_labels)
            str_val_labels = '[' + str_val_labels + ']'
            # print(f'Actual labels: [{str_val_labels}]')
            predictions = predict_model(model, valX)
            # print(f'Predicted labels: {predictions}')
            results = evaluate(val_labels, predictions)
            # print(results)

            arg_str = f'[{temp_n_bins}, {temp_word_size}, {temp_window_size}]'
            csv_writer.writerow(
                [arg_str, results[f1_label], results[recall_label], results[precision_label], str_val_labels,
                 predictions])

csv_file.close()

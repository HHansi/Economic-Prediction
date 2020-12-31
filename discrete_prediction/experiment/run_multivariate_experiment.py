# Created by Hansi at 11/23/2020

import numpy as np
import pandas as pd

from discrete_prediction.algo.common.evaluator import evaluate
from discrete_prediction.algo.common.label_generator import count_labels
from discrete_prediction.algo.common.multivariate_processor import create_split, format_training_data, \
    format_testing_data
from discrete_prediction.algo.saxvsm import train_multivariate_model, predict_model
from discrete_prediction.args import data_file_path, multivariate_data_processing_args, saxvsm_args
from discrete_prediction.algo.common.data_preprocessor import normalise_train_data, normalise_data

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

# format data
# train, val = create_split(filtered_df, args=multivariate_data_processing_args)
trainX, train_labels, valX, val_labels = format_training_data(train, val, args=multivariate_data_processing_args)

# calculate label distribution
unique_elements, counts_elements = count_labels(train_labels)
print(f"Training data set label distribution: \n {np.asarray((unique_elements, counts_elements))}")
unique_elements, counts_elements = count_labels(val_labels)
print(f"Testing data set label distribution: \n {np.asarray((unique_elements, counts_elements))}")

# print(np.where(np.var(trainX, axis=1) == 0)[0])

# train model
model = train_multivariate_model(trainX, train_labels, args=saxvsm_args)

# validate model
str_val_labels = ' '.join(str(x) for x in val_labels)
print(f'Actual labels: [{str_val_labels}]')
predictions = predict_model(model, valX)
print(f'Predicted labels: {predictions}')
results = evaluate(val_labels, predictions)
print(results)

# test predictions
# Here, we use the last time series which can extract from data file
test_df = filtered_df.iloc[-multivariate_data_processing_args['train_series_length']:, :]
# test_df = filtered_df.iloc[:multivariate_data_processing_args['train_series_length'], :]
test_data = test_df.to_numpy()

# normalise data
if multivariate_data_processing_args['normalise']:
    test_df = normalise_data(test_data)

# process and format test data
testX = format_testing_data(test_df, args=multivariate_data_processing_args)
test_predictions = predict_model(model, testX)
print(f'Test predictions: {test_predictions}')

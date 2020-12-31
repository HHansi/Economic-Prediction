# Created by Hansi at 11/22/2020
from discrete_prediction.algo.common.evaluator import evaluate
from discrete_prediction.algo.saxvsm import predict_model, train_model
from discrete_prediction.args import data_file_path, univariate_data_processing_args, saxvsm_args
from discrete_prediction.algo.common.univariate_processor import create_split, format_data

# load, process and format data
train, val = create_split(data_file_path, args=univariate_data_processing_args)
trainX, train_targs, valX, val_targs = format_data(train, val, args=univariate_data_processing_args)


# train model
model = train_model(trainX, train_targs, args=saxvsm_args)

# validate model
# print(f'Actual labels: {val_targs}')
str_val_labels = ' '.join(str(x) for x in val_targs)
print(f'Actual labels: [{str_val_labels}]')
predictions = predict_model(model, valX)
print(f'Predicted labels: {predictions}')
results =  evaluate(val_targs, predictions)
print(results)



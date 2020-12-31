# Created by Hansi at 12/30/2020
import pandas as pd

from discrete_prediction.algo.common.evaluator import evaluate
from discrete_prediction.algo.common.label_generator import discretize_series

data_file_path = "../../results/ARIMA/GDP_CVM-5.tsv"

df = pd.read_csv(data_file_path, delimiter='\t')
originals = df['Original'].dropna()
predictions = df['Prediction']

# remove future predictions
n = len(originals)
predictions = predictions[:n]

label_originals = discretize_series(originals)
label_predictions = discretize_series(predictions)

print(f'original: {label_originals}')
print(f'prediction: {label_predictions}')

results = evaluate(label_originals, label_predictions, average='weighted')
print(results)

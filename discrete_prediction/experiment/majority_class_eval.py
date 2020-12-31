# Created by Hansi at 12/1/2020
import numpy as np

from discrete_prediction.algo.common.evaluator import evaluate

actuals = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0]

majority_label = max(set(actuals), key=actuals.count)
preds = np.empty(len(actuals))
preds.fill(int(majority_label))
print(preds)

results = evaluate(actuals, preds)
print(results)




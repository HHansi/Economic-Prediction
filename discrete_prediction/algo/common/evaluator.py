# Created by Hansi at 11/22/2020
from sklearn import metrics

from discrete_prediction.args import f1_label, recall_label, precision_label


def evaluate(actuals, predictions, average = 'macro'):
    results = dict()
    f1 = metrics.f1_score(actuals, predictions, average=average)
    results[f1_label] = f1

    recall = metrics.recall_score(actuals, predictions, average=average)
    results[recall_label] = recall

    precision = metrics.precision_score(actuals, predictions, average=average)
    results[precision_label] = precision
    return results



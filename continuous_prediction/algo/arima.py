# Created by Hansi at 12/5/2020
from statsmodels.tsa.arima.model import ARIMA

from continuous_prediction.algo.common.util import grouped


def train_model(train, args):
    """
    Train ARIMA model

    :param train: list or data series
    :param args: json
    :return: object
        Trained model
    """
    model = ARIMA(train, order=(args['p'], args['d'], args['q']))
    model_fit = model.fit()
    return model_fit


def predict(model_fit, start, end):
    """
    Predict future values

    :param model_fit: object
        Trained model
    :param start: int
        Starting index of required predictions
    :param end: int
        Ending index of required predictions
    :return: list or series
        Predicted values
    """
    output = model_fit.predict(start=start, end=end, dynamic=True, typ='levels')
    return output


def rolling_forecast(train, test, data_args, arima_args):
    history = [x for x in train]
    predictions = list()
    future_n = data_args['future_n']
    for t in grouped(test, future_n):
        t = list(t)
        model_fit = train_model(history, arima_args)
        output = predict(model_fit, start=len(history), end=len(history) + future_n - 1)
        predictions.extend(output)
        history = history + t
    return predictions

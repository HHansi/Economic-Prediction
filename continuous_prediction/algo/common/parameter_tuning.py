# Created by Hansi at 12/6/2020

from statsmodels.tsa.stattools import adfuller


# if p-value>0.05 -> Accept NULL hypothesis -> not stationary
def adfuler_test(series):
    # Dickey–Fuller test:
    result = adfuller(series)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))


def shift_series(series, n):
    series_new = series - series.shift(n)
    return series_new


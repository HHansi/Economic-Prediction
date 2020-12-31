# Created by Hansi at 12/1/2020
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def normalise_train_data(data):
    return scaler.fit_transform(data)


def normalise_data(data):
    return scaler.transform(data)

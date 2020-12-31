# Created by Hansi at 11/22/2020
from pyts.classification import SAXVSM
from pyts.multivariate.classification import MultivariateClassifier


def train_model(trainX, train_targs, args):
    saxvsm = SAXVSM(n_bins=args['n_bins'], word_size=args['word_size'], window_size=args['window_size'],
                    use_idf=True,
                    smooth_idf=True,
                    sublinear_tf=True,
                    strategy='uniform')

    saxvsm.fit(trainX, train_targs)
    return saxvsm


def train_multivariate_model(trainX, train_labels, args):
    saxvsm = SAXVSM(n_bins=args['n_bins'], word_size=args['word_size'], window_size=args['window_size'],
                    use_idf=True,
                    smooth_idf=True,
                    sublinear_tf=True,
                    strategy='uniform')
    m_saxsvm = MultivariateClassifier(saxvsm)
    m_saxsvm.fit(trainX, train_labels)
    return m_saxsvm


def predict_model(model, valX):
    predictions = model.predict(valX)
    return predictions

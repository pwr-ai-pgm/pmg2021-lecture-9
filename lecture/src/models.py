import numpy as np
from sklearn.preprocessing import StandardScaler

from .blr import ConjugateBayesLinReg, FullConjugateBayesLinReg


def get_model_1(data):
    x = np.stack([
        data[0],
        data[0]**2,
        data[0]**3,
        data[0]**4
    ], axis=1)
    x = StandardScaler().fit_transform(x)
    model = FullConjugateBayesLinReg(n_features=x.shape[1])
    model.fit(x, data[1])

    return model, x


def get_model_2(data):
    x = np.stack([
        data[0],
        data[0] ** 2,
        data[0] ** 3,
    ], axis=1)
    x = StandardScaler().fit_transform(x)
    model = FullConjugateBayesLinReg(n_features=x.shape[1])
    model.fit(x, data[1])

    return model, x


def get_model_3(data):
    x = np.stack([
        data[0],
        data[0] ** 2,
        data[0] ** 3,
        data[0] ** 4
    ], axis=1)
    x = StandardScaler().fit_transform(x)
    model = ConjugateBayesLinReg(n_features=x.shape[1], alpha=1, lmbda=0.5)
    model.fit(x, data[1])

    return model, x

import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge


class ConjugateBayesLinReg:

    def __init__(self, n_features, alpha, lmbda):
        self.n_features = n_features
        self.alpha = alpha
        self.lmbda = lmbda
        self.mean = np.zeros(n_features)
        self.cov_inv = np.identity(n_features) * alpha
        self.cov = np.linalg.inv(self.cov_inv)
        self.intercept = 0

    def fit(self, x, y):
        # If x and y are singletons, then we coerce them to a batch of length 1
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)

        self.intercept = np.mean(y)
        y = y - self.intercept

        # Update the inverse covariance matrix (Bishop eq. 3.51)
        cov_inv = self.cov_inv + self.lmbda * x.T @ x

        # Update the mean vector (Bishop eq. 3.50)
        cov = np.linalg.inv(cov_inv)
        mean = cov @ (self.cov_inv @ self.mean + self.lmbda * y @ x)

        self.cov_inv = cov_inv
        self.cov = cov
        self.mean = mean

    def predict(self, x):
        '''Metoda zwraca rozkład predykcyjny zmiennej niezależnej na bazie bieżących parametrów modelu.'''
        # If x and y are singletons, then we coerce them to a batch of length 1
        x = np.atleast_2d(x)

        # Obtain the predictive mean (Bishop eq. 3.58)
        y_pred_mean = x @ self.mean + self.intercept

        # Obtain the predictive variance (Bishop eq. 3.59)
        # np.diagonal bo nie interesuje nas kowariancja między próbkami testowymi
        # alternatywnie iterować po `x` i zbierać w mean i var w wektory
        y_pred_var = 1 / self.lmbda + np.diagonal((x @ self.cov @ x.T))

        # Drop a dimension from the mean and variance in case x and y were singletons
        y_pred_mean = np.squeeze(y_pred_mean)
        y_pred_var = np.squeeze(y_pred_var)

        return stats.norm(loc=y_pred_mean, scale=y_pred_var ** .5)

    @property
    def weights_dist(self):
        return stats.multivariate_normal(mean=self.mean, cov=self.cov)


class FullConjugateBayesLinReg(ConjugateBayesLinReg):

    def __init__(
            self,
            n_features,
            n_iter=300,
            tol=1.e-3,
            alpha_shape=1.e-6,  # non-informative prior
            alpha_rate=1.e-6,
            lmbda_shape=2,  # mode=1 => standard normal posterior
            lmbda_rate=1,
            alpha_init=1,
            lmbda_init=1,
            fit_intercept=True
    ):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_shape = alpha_shape
        self.alpha_rate = alpha_rate
        self.lmbda_shape = lmbda_shape
        self.lmbda_rate = lmbda_rate
        self.alpha = alpha_init
        self.lmbda = lmbda_init
        self.fit_intercept = fit_intercept

        self.mean = np.zeros((n_features,))
        self.cov = np.eye(n_features) / self.alpha
        self.intercept = 0

    def fit(self, x, y):
        # If x and y are singletons, then we coerce them to a batch of length 1
        x = np.atleast_2d(x)
        y = np.atleast_1d(y)

        model = BayesianRidge(
            n_iter=self.n_iter,
            tol=self.tol,
            alpha_1=self.lmbda_shape,
            alpha_2=self.lmbda_rate,
            lambda_1=self.alpha_shape,
            lambda_2=self.alpha_rate,
            alpha_init=self.lmbda,
            lambda_init=self.alpha,
            fit_intercept=self.fit_intercept,
            normalize=False,
        )

        model.fit(x, y)

        self.mean = model.coef_
        self.cov = model.sigma_
        self.alpha = model.lambda_
        self.lmbda = model.alpha_
        self.intercept = model.intercept_

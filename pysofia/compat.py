from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from sklearn import base
from sklearn.externals import joblib
from .sofia_ml import svm_train, learner_type, loop_type, eta_type


class RankSVM(base.BaseEstimator):
    """ RankSVM model using stochastic gradient descent.
    TODO: does this fit intercept ?

    Parameters
    ----------
    alpha : float

    model : str, default='rank'
       

    max_iter : int, default=1000
       Number of stochastic gradient steps to take
    """

    def __init__(self, alpha=1., model='rank', max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = model

    def fit(self, X, y, query_id=None):
        n_samples, n_features = X.shape

        self.coef_ = svm_train(X, y, query_id, self.alpha, n_samples,
                               n_features, learner_type.sgd_svm,
                               loop_type.rank, eta_type.basic_eta,
                               max_iter=self.max_iter)
        return self

    def rank(self, X):
        order = np.argsort(X.dot(self.coef_))
        order_inv = np.zeros_like(order)
        order_inv[order] = np.arange(len(order))
        return order_inv

    # just so that GridSearchCV doesn't complain
    predict = rank

    def score(self, X, y):
        tau, _ = stats.kendalltau(X.dot(self.coef_), y)
        return np.abs(tau)


def _inner_fit(X, y, query_id, train, test, alpha):
    # aux method for joblib
    clf = RankSVM(alpha=alpha)
    if query_id is None:
        clf.fit(X[train], y[train])
    else:
        clf.fit(X[train], y[train], query_id[train])
    return clf.score(X[test], y[test])

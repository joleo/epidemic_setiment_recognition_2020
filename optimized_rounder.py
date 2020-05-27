# -*- coding: utf-8 -*-
"""
@Time ： 2020/3/14 9:37
@Auth ： joleo
@File ：optimized_rounder.py
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from functools import partial
import scipy as sp

class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    """

    def __init__(self):
        self.coef_ = 0

    def _macro_f1_score(y_true, y_pred, n_labels):
        # https://www.kaggle.com/corochann/fast-macro-f1-computation
        total_f1 = 0.
        for i in range(n_labels):
            yt = y_true == i
            yp = y_pred == i

            tp = np.sum(yt & yp)

            tpfp = np.sum(yp)
            tpfn = np.sum(yt)
            if tpfp == 0:
                print('[WARNING] F-score is ill-defined and being set to 0.0 in labels with no predicted samples.')
                precision = 0.
            else:
                precision = tp / tpfp
            if tpfn == 0:
                print(f'[ERROR] label not found in y_true...')
                recall = 0.
            else:
                recall = tp / tpfn
            if precision == 0. or recall == 0.:
                f1 = 0.
            else:
                f1 = 2 * precision * recall / (precision + recall)
            total_f1 += f1
        return total_f1 / n_labels

    def _f1_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients

        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2])

        return -f1_score(y, X_p, average='macro')

    def fit(self, X, y):
        """
        Optimize rounding thresholds

        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds

        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']

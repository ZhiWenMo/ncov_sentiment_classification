from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score


class OptimizedMacroF1Rounder:
    def __init__(self, num_labels):
        self.coef_ = 0
        self.num_labels = num_labels

    def _f1_loss(self, coef, X, y):
        X_p = np.copy(X)

        pred = np.argmax(X_p * coef, axis=1)
        macro_f1 = f1_score(y, pred, average="macro")
        return -macro_f1

    def fit(self, X, y):
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.1, 0.7, 0.2]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        X_p = X_p * coef
        X_p = np.argmax(X_p, axis=1)

        return X_p

    def predict_proba(self, X, coef):
        X_p = np.copy(X)
        X_p = X_p * coef

        return X_p

    def coefficients(self):
        return self.coef_.x
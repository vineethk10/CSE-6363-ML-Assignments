"""
CSE 6363-007: Machine Learning Assignment 5
Name: Ananthula, Vineeth Kumar. UTA ID: 1001953922

Random Forest Implementation
"""

import copy
import numpy as np
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

class RandomForest:

    def __init__(self,
                 base_learner,
                 num_trees,
                 min_features=2020):
        
        np.random.seed(min_features)
        self.base_learner = base_learner
        self.num_trees = num_trees
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.num_trees)]

    def _get_bootstrap_dataset(self, X, y, i):
        X_arr, y_arr=np.array(X), np.array(y)
        idx=np.random.choice(X_arr.shape[0], X_arr.shape[0], replace=True)
        X_bootstrap, y_bootstrap = X.iloc[idx, :], y.iloc[idx]
        self._estimators[i].fit(X_bootstrap, y_bootstrap)

    def fit(self, X, y):
        self.labels=list(set(y))
        part_bootstrap=partial(self._get_bootstrap_dataset, X, y)
        pool=ThreadPool(self.num_trees)
        pool.map(part_bootstrap, list(range(self.num_trees)))
        pool.close()
        pool.join()
        return self

    def predict(self, X):
        N = X.shape[0]
        y_pred = np.zeros(N)
        preds=np.array([estimator.predict(X) for estimator in self._estimators])
        pred_proba=np.array([(preds == label).mean(axis=0) for label in self.labels]).T
        y_pred=np.array(self.labels)[np.argmax(pred_proba, axis=1)]
        return y_pred


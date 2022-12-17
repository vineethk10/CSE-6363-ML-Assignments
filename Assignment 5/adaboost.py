"""
CSE 6363-007: Machine Learning Assignment 5
Name: Ananthula, Vineeth Kumar. UTA ID: 1001953922

Adaboost Implementation
"""
import copy
import numpy as np


class Adaboost:

    def __init__(self,
                 weak_learner,
                 num_learners,
                 seed=2020):
 
        np.random.seed(seed)
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self._estimators = [copy.deepcopy(self.weak_learner) for _ in range(self.num_learners)]
        self._alphas = [1 for _ in range(num_learners)]

    def fit(self, X, y):
        L=y.shape[0]
        w=np.ones(L)/L
        for i in range(self.num_learners):
            self._estimators[i].fit(X, y, weightSample=w)
            pred=self._estimators[i].predict(X)
            err=np.sum((pred!=y)*w)
            alpha=0.5*np.log((1-err)/err)
            self._alphas[i]=alpha
            w*=np.exp(-alpha*np.array(y)*pred) + 1e-4
            w/=2*np.sqrt(err*(1-err))
        return self

    def predict(self, X):
        N = X.shape[0]
        y_pred = np.zeros(N)
        for i in range(self.num_learners):
            y_pred += self._alphas[i]*self._estimators[i].predict(X)           
        y_pred=np.sign(y_pred)
        return y_pred

"""
CSE 6363-007: Machine Learning Assignment 5
Name: Ananthula, Vineeth Kumar. UTA ID: 1001953922

Decision Tree Implementation
"""
import numpy as np

class DecisionTree:

    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_leaf,
                 sample_feature=False):
        if criterion == 'ratioInfoGain':
            self.criterion = self._information_gain_ratio
        elif criterion == 'entropy':
            self.criterion = self._information_gain
        elif criterion == 'gini':
            self.criterion = self._gini_purification
        else:
            raise Exception('Criterion should be ratioInfoGain or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature

    def fit(self, X, y, weightSample=None):
        if weightSample is None:
            weightSample = np.ones(X.shape[0]) / X.shape[0]
        else:
            weightSample = np.array(weightSample) / np.sum(weightSample)

        featureNames = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, featureNames, depth=1, weightSample=weightSample)
        return self

    @staticmethod
    def entropy(y, weightSample=None):
        entropy = 0.0
        num=y.shape[0]#number of labels
        labelCounter={}
        for i in range(num):
            if y[i] not in labelCounter.keys():
                labelCounter[y[i]]=0
            labelCounter[y[i]]+=weightSample[i]
        for key in labelCounter:
            prob=float(labelCounter[key])/float(np.sum(weightSample))
            entropy-=prob*np.log2(prob)
        return entropy

    def _information_gain(self, X, y, index, weightSample=None):

        infoGain = 0
        oldCost=self.entropy(y, weightSample)
        uniqueValues=np.unique(X[:,index])
        newCost=0.0
        for value in uniqueValues:
            X_sub,Y_sub, subSampleWeight=self._split_dataset(X, y, index, value, weightSample)
            prob=np.sum(subSampleWeight)/float(np.sum(weightSample))
            newCost+=prob*self.entropy(Y_sub, subSampleWeight)
        infoGain=oldCost-newCost
        return infoGain

    def _information_gain_ratio(self, X, y, index, weightSample=None):
        ratioInfoGain = 0
        informationSplit = 0.0
        oldCost=self.entropy(y, weightSample)
        uniqueValues=np.unique(X[:,index])
        newCost=0.0
        informationSplit=0.0
        for value in uniqueValues:
            X_sub,Y_sub, subSampleWeight=self._split_dataset(X, y, index, value, weightSample)
            prob=np.sum(subSampleWeight)/float(np.sum(weightSample))
            newCost+=prob*self.entropy(Y_sub, subSampleWeight)
            informationSplit-=prob*np.log2(prob)
        if informationSplit==0.0:
            pass
        else:
            infoGain=oldCost-newCost
            ratioInfoGain=infoGain/informationSplit
        return ratioInfoGain

    @staticmethod
    def gini_impurity(y, weightSample=None):
        gini = 1
        num=y.shape[0]
        labelCounter={}
        for i in range(num):
            if y[i] not in labelCounter.keys():
                labelCounter[y[i]]=0
            labelCounter[y[i]]+=weightSample[i]
        for key in labelCounter:
            prob=float(labelCounter[key])/float(np.sum(weightSample))
            gini -= prob ** 2
        return gini

    def _gini_purification(self, X, y, index, weightSample=None):
        newImpurity = 0
        oldCost=self.gini_impurity(y, weightSample)
        uniqueValues=np.unique(X[:,index])
        newCost=0.0 
        for value in uniqueValues:
            X_sub,Y_sub, subSampleWeight=self._split_dataset(X, y, index, value, weightSample)
            prob=np.sum(subSampleWeight)/float(np.sum(weightSample))
            newCost+=prob*self.gini_impurity(Y_sub, subSampleWeight)
        newImpurity=oldCost-newCost 
        return newImpurity

    def _split_dataset(self, X, y, index, value, weightSample=None):
        ret=[]
        featVec=X[:,index]
        X=X[:,[i for i in range(X.shape[1]) if i!=index ]]
        for i in range(len(featVec)):
            if featVec[i]==value:
                ret.append(i)
        X_sub = X[ret,:]
        Y_sub = y[ret]
        subSampleWeight=weightSample[ret]
        return X_sub, Y_sub, subSampleWeight

    def _choose_best_feature(self, X, y, weightSample=None):
        bestFeature = 0
        nFeatures = X.shape[1]
        if self.sample_feature:
            maxFeatures=max(1, min(nFeatures, int(np.round(np.sqrt(nFeatures)))))
            newFeatures=np.random.choice(nFeatures, maxFeatures, replace=False)
            new_X=X[:, newFeatures]
        else:
            new_X=X
        n_newFeatures=new_X.shape[1]
        best_gain_cost=0.0
        for i in range(n_newFeatures):
            infoGain_cost=self.criterion(new_X,y,i,weightSample)           
            if infoGain_cost > best_gain_cost:
                best_gain_cost=infoGain_cost
                bestFeature=i                
        return bestFeature

    @staticmethod
    def majority_vote(y, weightSample=None):
        if weightSample is None:
            weightSample = np.ones(y.shape[0]) / y.shape[0]
        majorityLabel = y[0]
        dict_num={}
        for i in range(y.shape[0]):
            if y[i] not in dict_num.keys():
                dict_num[y[i]]=weightSample[i]
            else:
                dict_num[y[i]] += weightSample[i]
        majorityLabel=max(dict_num, key=dict_num.get)
        return majorityLabel

    def _build_tree(self, X, y, featureNames, depth, weightSample=None):
        mytree = dict()
        if len(featureNames)==0 or len(np.unique(y))==1 or depth >= self.max_depth or len(X) <= self.min_samples_leaf: 
            return self.majority_vote(y, weightSample)
        bestFeature=self._choose_best_feature(X, y, weightSample)
        bestFeatureName = featureNames[bestFeature]
        featureNames=featureNames[:]
        featureNames.remove(bestFeatureName)
        mytree={bestFeatureName:{}}
        uniqueValues=np.unique(X[:, bestFeature])
        for value in uniqueValues:
            X_sub, Y_sub, subSampleWeight = self._split_dataset(X, y, bestFeature, value, weightSample)
            mytree[bestFeatureName][value]=self._build_tree(X_sub, Y_sub, featureNames, depth+1, subSampleWeight)
        return mytree

    def predict(self, X):
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        def _classify(tree, x):
            featureName=list(tree.keys())[0] 
            secondDict=tree[featureName]            
            key=x.loc[featureName]
            if key not in secondDict:
                key=np.random.choice(list(secondDict.keys()))
            valueOfKey=secondDict[key]
            if isinstance(valueOfKey,dict):
                label=_classify(valueOfKey,x)
            else:
                label=valueOfKey
            return label
        if len(X.shape)==1:
            return _classify(self._tree,X)
        else:
            results=[]
            for i in range(X.shape[0]):
                results.append(_classify(self._tree, X.iloc[i, :]))
            return np.array(results)
    def show(self):
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)
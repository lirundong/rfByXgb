# -*- coding: utf-8 -*-
# Rundong Li, UESTC

import xgboost as xgb
import pandas as pd
import numpy as np
from numpy.random import uniform
from numpy import floor, ceil

# default para dicts for pandas.DataFrame to XGBoost.DMatrix:
PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64': 'int',
                       'uint8': 'int', 'uint16': 'int', 'uint32': 'int', 'uint64': 'int',
                       'float16': 'float', 'float32': 'float', 'float64': 'float',
                       'bool': 'i'}


# default para dicts for XGBoost.general:
DEFAULT_XGB_GENERAL_PARA_DIC = {'booster':'gbtree', 'slient':1, 'nthread':1, 'num_pbuffer':None, 'num_feature':None,
                                'objective':'binary:logistic'}

# default para dicts for XGBoost.treeBooster:
DEFAULT_XGB_TREE_BOOSTER_PARA_DIC = {'eta':0.3, 'gamma':0, 'max_depth':6, 'min_child_weight':1,'max_delta_step':0,
                                    'subsample':1, 'colsample_bytree':1, 'colsample_bylevel':1, 'lambda':1, 'alpha':0,
                                    'tree_method':'auto','sketch_eps':0.03}

# default para dicts for randomForest:
DEFAULT_RF_PARA_DIC = {'n_estimators': 10, 'criterion': 'gini', 'max_depth': None,
                    'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0,
                    'max_features': 'auto', 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': False,
                    'n_jobs': 1, 'random_state': None, 'verbose': 0, 'warm_start': False, 'class_weight': None}

# default para dicts for XGBoost.boost.train:
DEFAULT_XGB_TRAIN_PARA_DIC = {'params': None, 'dtrain': None, 'num_boost_round': 10, 'evals': None,
                              'obj': None, 'feval': None, 'maximize': False, 'early_stopping_rounds': None,
                              'evals_result': None, 'verbose_eval': True, 'learning_rates': None, 'xgb_model': None}

# parameter map from RandomForestClassifier to XGBoost:
RF_TO_XGB_PARA_DIC = {'n_estimators': 'num_boost_round', 'criterion': None, 'max_features':'num_feature',
                      'max_depth':'max_depth', 'min_samples_split':None, 'min_samples_leaf':None,
                      'min_weight_fraction_leaf':None,'max_leaf_nodes':None, 'bootstrap':None, 'oob_score':None,
                      'n_jobs':'nthread', 'random_state':None,'verbose':None, 'warm_start':None, 'class_weight':None}

# parameters can be constructed:
CAN_CON_PARA = ['bootstrap']

# new parameters:
# -----
#  oneHotThre: [float, default = 0.005]
#              threshould when encode categorical feature to one-hot code
#              if (currentCateFeature.num() / allInstance.num() < oneHotThre)
#              then we think this categorical candidate is noise and ignore it;
#  bootStrapRounds: [int, default = 10]
#
NEW_PARA_DIC = {'oneHotThre':0.005, 'bootStrapRounds':10, 'bootStrapRato':0.1, 'bootStrapEqu':True}


# Functions for further using -- WAITING FIX
def findCateFeat(seriesIn, cateFeatList, cateColList, cateColSure=False):
    """
        Function to enumerate all categorical feature candidates in a
    pandas.DataFrame into a list for further One-Hot encode.
    ==========
    :param seriesIn: Type: pandas.DataFrame
            The DataFrame you're dealing with.
            It should contains as least one column which made up
        with CATEGORICAL candidates(not int, float or bool).

    :param cateFeatList: Type: list

    :return: NULL
    """
    if not cateColSure:
        # don't know which columns is categorical, judge by *.dtype.name
        if seriesIn.dtype.name not in PANDAS_DTYPE_MAPPER.keys():
            cateFeatList.extend({}.fromkeys(seriesIn.values).keys())
            if seriesIn.name not in cateColList:
                cateColList.append(seriesIn.name)
    else:
        if len(cateColList) == 0:
            raise ValueError('If categorical column is sure, param "cateColList" can not be empty.')
        elif (seriesIn.dtype.name not in PANDAS_DTYPE_MAPPER) and (seriesIn.dtype.name in cateColList):
            # return
        # else:
            cateFeatList.extend({}.fromkeys(seriesIn.values).keys())


class Learner:
    def __init__(self, paraDic, xgbParaDic=None):
        """
        Init the Learner class.

        :param paraDic: dict of parameters
        Parameters
        complete same as sklearn.ensemble.RandomForestClassifier
    ----------
        n_estimators : integer, optional (default=10)
            The number of trees in the forest.
        criterion : string, optional (default="gini")
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "entropy" for the information gain.
            Note: this parameter is tree-specific.
        max_features : int, float, string or None, optional (default="auto")
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a percentage and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.
            Note: this parameter is tree-specific.
        max_depth : integer or None, optional (default=None)
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
            Ignored if ``max_leaf_nodes`` is not None.
            Note: this parameter is tree-specific.
        min_samples_split : integer, optional (default=2)
            The minimum number of samples required to split an internal node.
            Note: this parameter is tree-specific.
        min_samples_leaf : integer, optional (default=1)
            The minimum number of samples in newly created leaves.  A split is
            discarded if after the split, one of the leaves would contain less then
            ``min_samples_leaf`` samples.
            Note: this parameter is tree-specific.
        min_weight_fraction_leaf : float, optional (default=0.)
            The minimum weighted fraction of the input samples required to be at a
            leaf node.
            Note: this parameter is tree-specific.
        max_leaf_nodes : int or None, optional (default=None)
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.
            If not None then ``max_depth`` will be ignored.
            Note: this parameter is tree-specific.
        bootstrap : boolean, optional (default=True)
            Whether bootstrap samples are used when building trees.
        oob_score : bool
            Whether to use out-of-bag samples to estimate
            the generalization error.
        n_jobs : integer, optional (default=1)
            The number of jobs to run in parallel for both `fit` and `predict`.
            If -1, then the number of jobs is set to the number of cores.
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        verbose : int, optional (default=0)
            Controls the verbosity of the tree building process.
        warm_start : bool, optional (default=False)
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit a whole
            new forest.
        class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``
            The "balanced_subsample" mode is the same as "balanced" except that weights are
            computed based on the bootstrap sample for every tree grown.
            For multi-output, the weights of each column of y will be multiplied.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.

        """

        # PARAMETERS
        self.paraDic = NEW_PARA_DIC.copy()
        self.xgbSetPara = DEFAULT_XGB_TRAIN_PARA_DIC.copy()
        self.xgbTrainPara = DEFAULT_XGB_GENERAL_PARA_DIC.copy()
        for keys in DEFAULT_XGB_TREE_BOOSTER_PARA_DIC.keys():
            self.xgbTrainPara.update({'bst:'+keys : DEFAULT_XGB_TREE_BOOSTER_PARA_DIC.get(keys)})

        self.paraNotInXGB = {}
        for currentKey in paraDic.keys():
            if RF_TO_XGB_PARA_DIC.get(currentKey) is not None:
                if RF_TO_XGB_PARA_DIC.get(currentKey) in self.xgbSetPara.keys():
                    # current feature is XGBoost set para
                    self.xgbSetPara[RF_TO_XGB_PARA_DIC.get(currentKey)] = paraDic[currentKey]
                elif RF_TO_XGB_PARA_DIC.get(currentKey) in self.xgbTrainPara.keys():
                    # current feature is general train feature
                    self.xgbTrainPara[RF_TO_XGB_PARA_DIC.get(currentKey)] = paraDic[currentKey]
                else:
                    # current feature is boost feature
                    self.xgbTrainPara['bst:'+RF_TO_XGB_PARA_DIC.get(currentKey)] = paraDic[currentKey]
            elif currentKey in CAN_CON_PARA or NEW_PARA_DIC:
                self.paraDic.update({currentKey: paraDic[currentKey]})
            else:
                # this feature should be constructed manually
                self.paraNotInXGB[currentKey] = paraDic[currentKey]
        for keys in self.paraNotInXGB.keys():
            print "WARNING: parameter %s can't constructed.\n" % keys

        # store XGBoost parameters:
        # self.xgbTrainPara.update(xgbParaDic)
        if xgbParaDic is not None:
            for item in xgbParaDic.items():
                if item[0] in self.xgbSetPara.keys():
                    # current feature is XGBoost set para
                    self.xgbSetPara[item[0]] = item[1]
                elif item[0] in self.xgbTrainPara.keys():
                    # current feature is general train feature
                    self.xgbTrainPara[item[0]] = item[1]
                else:
                    # current feature is boost feature
                    self.xgbTrainPara['bst:' + item[0]] = item[1]

        # clear None parameters
        for key in self.xgbSetPara.keys():
            if self.xgbSetPara[key] is None:
                del self.xgbSetPara[key]

        for key in self.xgbTrainPara.keys():
            if self.xgbTrainPara[key] is None:
                del self.xgbTrainPara[key]

        # TRAIN AND TEST DATA STORAGE
        self.trainLabel = pd.DataFrame()
        self.trainFeat = pd.DataFrame()

        self.cateFeatColList = []  # columns which is categorical
        self.cateFeatList = []  # candidates of categorical feats
        self.numFeatColList = []
        self.oneHotCol = []  # one-Hot code column name

        self.bsBoosterList = []  # store Boosters of fit result

    def resetPara(self, newParaDic):
        """
        reset parameter via a new parameter dict

        :param newParaDic: a new parameter dict
        :return: None
        """
        for currentKey in newParaDic.keys():
            # self.paraDic[currentKey] = newParaDic.get(currentKey)
            if currentKey in self.paraDic.keys():
                self.paraDic[currentKey] = newParaDic.get(currentKey)
            else:
                print 'Invalid parameter when reset Parameters: %s' % currentKey
                exit(-1)

    def fit(self, X, y):
        """
        fit X = [labels, features] to Y = [features]

        :param X: 2-dim Pandas.DataFrame
            [itemNum * (1 + featureNum)]: [labels, features]
            feature can be int, float, bool or categorical info
        :param Y: 2-dim Pandas.DataFrame
            [itemNum * featureNum]: [features]
        :return: non
        """

        # DATA PRE-PROCESS:
        if len(X) != len(y):
            raise ValueError('train feature and label must get same length!')

        if type(X) != pd.core.frame.DataFrame:
            raise TypeError('train feature X must be a pandas.DataFrame!')
        else:
            trainFeature = X
        numTrainIns = len(X.index)

        if type(y) != pd.core.frame.DataFrame:
            self.trainLabel = pd.DataFrame(y, columns=['label'], index = X.index)
        else:
            self.trainLabel = y

        # if categorical, encode to one-hot code
        # note threshold rate parameter: oneHotThre
        trainFeatureCate = pd.DataFrame()

        # for column in trainFeature.columns:
        #     if trainFeature[column].dtype.name not in PANDAS_DTYPE_MAPPER:
        #         self.cateFeatColList.append(column)
        #         for featName in list(set(trainFeature[column])):
        #             if ((float(sum(trainFeature.ix[:,column] == featName)) / float(numTrainIns)) >= self.paraDic.get('oneHotThre'))\
        #                     and (featName not in self.cateFeatList):
        #                 # de-noise: if (currentCateFeature.num() / allInstance.num() < oneHotThre)
        #                 # then we think this categorical candidate is noise and ignore it;
        #                 self.cateFeatList.append(featName)
        #                 print 'get new categorical feat: %s' % featName
        trainFeature.apply(findCateFeat, args=(self.cateFeatColList, self.cateFeatList))

        self.numFeatColList = trainFeature.columns.difference(self.cateFeatColList)
        trainFeatureNum = trainFeature.ix[:,self.numFeatColList]
        # testFeatureNum = testFeature.ix[:,self.numFeatColList]

        # one-Hot code column name:
        self.oneHotCol = ['oneHot_'+str(num) for num in range(len(self.cateFeatList))]

        # dealing with train categorical features:
        currentNum = 1
        for idx in trainFeature.index:
            currentCateList = list(trainFeature.ix[idx, self.cateFeatColList])
            currentOneHot = [cateFeat in currentCateList for cateFeat in self.cateFeatList]
            trainFeatureCate = trainFeatureCate.append(pd.DataFrame(currentOneHot, columns=[idx],
                                                                    index=self.oneHotCol).T)
            print 'One-hot coding... (%d/%d)' % (currentNum, len(trainFeature))
            currentNum += 1
        self.trainFeat = pd.concat([trainFeatureNum, trainFeatureCate], axis=1)

        # let's learn via XGBoost
        if self.paraDic.get('bootstrap'):
            # use bootstrap to sample
            # seed()
            for bsRound in range(self.paraDic.get('bootStrapRounds')):
                # boot strap for bootStrapRounds rounds
                if self.paraDic.get('bootStrapEqu'):
                    # negInstans.num() = posInstance.num()
                    postiveIdx = self.trainLabel[self.trainLabel['label'] == 1].index
                    negativeIdx = self.trainLabel.index.difference(postiveIdx)
                    currentPosIdx = postiveIdx[
                        floor(uniform(size=(1, ceil(len(postiveIdx) * self.paraDic.get('bootStrapRato')))) * len(postiveIdx)).astype(int).tolist()]
                    currentNegIdx = negativeIdx[
                        floor(uniform(size=(1, ceil(len(negativeIdx) * self.paraDic.get('bootStrapRato')))) * len(negativeIdx)).astype(int).tolist()]
                    posInstance = self.trainFeat.ix[currentPosIdx, :]
                    negInstance = self.trainFeat.ix[currentNegIdx, :]
                    # store into XGBoost
                    currentTrainDF = posInstance.append(negInstance)
                    currentTrainLabel = [True] * len(currentPosIdx) + [False] * len(currentNegIdx)
                else:
                    # don't care if pos instance equal to neg instance
                    sampleTrainNum = ceil(float(numTrainIns) * self.paraDic.get('bootStrapRato'))
                    sampleTrainIdx = self.trainFeat.index[floor(uniform(size=(1, sampleTrainNum)) * numTrainIns).astype(int).tolist()]
                    currentTrainDF = self.trainFeat.ix[sampleTrainIdx]
                    currentTrainLabel = self.trainLabel.ix[sampleTrainIdx]
                # learn and dump a Booster per round
                currentTrainDM = xgb.DMatrix(currentTrainDF, label=currentTrainLabel)
                currentBooster = xgb.train(self.xgbTrainPara, currentTrainDM, **self.xgbSetPara)
                print 'Training:\tbootStrap Round #\t%d\n' % bsRound
                self.bsBoosterList.append(currentBooster)
        else:
            # don't bootstrap
            # load trainFeature to xgboost
            trainDMat = xgb.DMatrix(self.trainFeat, label=self.trainLabel)
            # self.testDMat = xgb.DMatrix(self.testFeat)
            currentBooster = xgb.train(self.xgbTrainPara, trainDMat, **self.xgbSetPara)
            print 'NO BOOTSTRAP:\ttrain finished.\n'
            self.bsBoosterList.append(currentBooster)

    def predict(self, X):
        # one-hot coding
        predictFeature = X
        predictFeatureCate = pd.DataFrame()
        for idx in predictFeature.index:
            currentCateList = list(predictFeature.ix[idx, self.cateFeatColList])
            currentOneHot = [cateFeat in currentCateList for cateFeat in self.cateFeatList]
            predictFeatureCate = predictFeatureCate.append(pd.DataFrame(currentOneHot, columns=[idx],
                                                                    index=self.oneHotCol).T)
        predictFeatureNum = predictFeature.ix[:, self.numFeatColList]
        predictFeat = pd.concat([predictFeatureNum, predictFeatureCate], axis=1)
        predictDM = xgb.DMatrix(predictFeat)
        self.finalPrediction = np.zeros((1,len(X)))
        if self.paraDic.get('bootstrap'):
            # oops, we should predict multipal times
            weightPerBoost = 1.0 / float(len(self.bsBoosterList))
            for booster in self.bsBoosterList:
                currentPrediction = booster.predict(predictDM)
                # print currentPrediction
                self.finalPrediction += currentPrediction * weightPerBoost
        else:
            self.finalPrediction = self.bsBoosterList[0].predict(predictDM)
        print self.finalPrediction
        return pd.DataFrame(self.finalPrediction.T, index=X.index)
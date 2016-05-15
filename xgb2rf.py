# -*- coding: utf-8 -*-
# Rundong Li, UESTC

import xgboost as xgb
import pandas as pd
import numpy as np
# from numpy.random import uniform
from numpy import ceil  # , floor

# default para dicts for pandas.DataFrame to XGBoost.DMatrix:
PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64': 'int',
                       'uint8': 'int', 'uint16': 'int', 'uint32': 'int', 'uint64': 'int',
                       'float16': 'float', 'float32': 'float', 'float64': 'float',
                       'bool': 'i'}


# default para dicts for XGBoost.general:
DEFAULT_XGB_GENERAL_PARA_DIC = {'booster': 'gbtree', 'slient': 1, 'nthread': 1, 'num_pbuffer': None,
                                'num_feature': None, 'objective': 'binary:logistic'}

# default para dicts for XGBoost.treeBooster:
DEFAULT_XGB_TREE_BOOSTER_PARA_DIC = {'eta': 0.3, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1,
                                     'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1,
                                     'colsample_bylevel': 1, 'lambda': 1, 'alpha': 0, 'tree_method': 'auto',
                                     'sketch_eps': 0.03}

# default para dicts for randomForest:
DEFAULT_RF_PARA_DIC = {'n_estimators': 10, 'criterion': 'gini', 'max_depth': None,
                       'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0,
                       'max_features': 'auto', 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': False,
                       'n_jobs': 1, 'random_state': None, 'verbose': 0, 'warm_start': False, 'class_weight': None}

# default para dicts for XGBoost.boost.train:
DEFAULT_XGB_TRAIN_PARA_DIC = {'num_boost_round': 10, 'evals': None,  # 'params': None, 'dtrain': None,
                              'obj': None, 'feval': None, 'maximize': False, 'early_stopping_rounds': None,
                              'evals_result': None, 'verbose_eval': True, 'learning_rates': None, 'xgb_model': None}

# parameter map from RandomForestClassifier to XGBoost:
RF_TO_XGB_PARA_DIC = {'n_estimators': 'num_boost_round', 'criterion': None, 'max_features': 'num_feature',
                      'max_depth': 'max_depth', 'min_samples_split': None, 'min_samples_leaf': None, 'oob_score': None,
                      'min_weight_fraction_leaf': None, 'max_leaf_nodes': None, 'bootstrap': None, 'class_weight': None,
                      'n_jobs': 'nthread', 'random_state': None, 'verbose': None, 'warm_start': None}

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
NEW_PARA_DIC = {'oneHotThre': 0.005, 'bootStrapRounds': 10, 'bootStrapRato': 0.1, 'bootStrapEqu': True}


# Functions for further using -- WAITING FIX
def find_cate_feat(series_in, cate_feat_list, cate_col_list, one_hot_thre=0.005, cate_col_sure=False):
    """
        Function to enumerate all categorical feature candidates in a
    pandas.DataFrame into a list for further One-Hot encode.
    ==========
    :param series_in: Type: pandas.DataFrame
            The DataFrame you're dealing with.
            It should contains as least one column which made up
        with CATEGORICAL candidates(not int, float or bool).

    :param cate_feat_list: Type: list

    :param cate_col_list:

    :param cate_col_sure:

    :return: NULL
    """
    if not cate_col_sure:
        # don't know which columns is categorical, judge by *.dtype.name
        if series_in.dtype.name not in PANDAS_DTYPE_MAPPER.keys():
            cate_feat_list.extend({}.fromkeys(series_in.values).keys())
            if series_in.name not in cate_col_list:
                cate_col_list.append(series_in.name)
    else:
        if len(cate_col_list) == 0:
            raise ValueError('If categorical column is sure, param "cate_col_list" can not be empty.')
        elif (series_in.dtype.name not in PANDAS_DTYPE_MAPPER) and (series_in.dtype.name in cate_col_list):
            cate_feat_list.extend({}.fromkeys(series_in.values).keys())


def one_hot_encode(row_in, cate_feat_list):
    return cate_feat_list.apply(lambda x: x in row_in.values)


class Learner:
    def __init__(self, para_dict, xgb_para_dict=None):
        """
        Init the Learner class.

        :param para_dict: dict of parameters
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
        self.para_dict = NEW_PARA_DIC.copy()
        self.xgbSetPara = DEFAULT_XGB_TRAIN_PARA_DIC.copy()
        self.xgbTrainPara = DEFAULT_XGB_GENERAL_PARA_DIC.copy()
        for keys in DEFAULT_XGB_TREE_BOOSTER_PARA_DIC.keys():
            self.xgbTrainPara.update({'bst:'+keys: DEFAULT_XGB_TREE_BOOSTER_PARA_DIC.get(keys)})

        self.paraNotInXGB = {}
        for currentKey in para_dict.keys():
            if RF_TO_XGB_PARA_DIC.get(currentKey) is not None:
                if RF_TO_XGB_PARA_DIC.get(currentKey) in self.xgbSetPara.keys():
                    # current feature is XGBoost set para
                    self.xgbSetPara[RF_TO_XGB_PARA_DIC.get(currentKey)] = para_dict[currentKey]
                elif RF_TO_XGB_PARA_DIC.get(currentKey) in self.xgbTrainPara.keys():
                    # current feature is general train feature
                    self.xgbTrainPara[RF_TO_XGB_PARA_DIC.get(currentKey)] = para_dict[currentKey]
                else:
                    # current feature is boost feature
                    self.xgbTrainPara['bst:'+RF_TO_XGB_PARA_DIC.get(currentKey)] = para_dict[currentKey]
            elif currentKey in CAN_CON_PARA or NEW_PARA_DIC:
                self.para_dict.update({currentKey: para_dict[currentKey]})
            else:
                # this feature should be constructed manually
                self.paraNotInXGB[currentKey] = para_dict[currentKey]
        for keys in self.paraNotInXGB.keys():
            print "WARNING: parameter %s can't constructed.\n" % keys

        # store XGBoost parameters:
        # self.xgbTrainPara.update(xgb_para_dict)
        if xgb_para_dict is not None:
            for item in xgb_para_dict.items():
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

    def reset_para(self, new_para_dict):
        """
        reset parameter via a new parameter dict

        :param new_para_dict: a new parameter dict
        :return: None
        """
        for currentKey in new_para_dict.keys():
            # self.paraDic[currentKey] = new_para_dict.get(currentKey)
            if currentKey in self.paraDic.keys():
                self.paraDic[currentKey] = new_para_dict.get(currentKey)
            else:
                print 'Invalid parameter when reset Parameters: %s' % currentKey
                exit(-1)

    def fit(self, X, y):
        """
        fit X = [labels, features] to y = [features]

        :param X: 2-dim pandas.DataFrame
            [itemNum * (1 + featureNum)]: [labels, features]
            feature can be int, float, bool or categorical info
        :param y: pandas.Series
            [itemNum * featureNum]: [features]
        :return: NULL
        """

        # DATA PRE-PROCESS:
        if len(X) != len(y):
            raise ValueError('train feature and label must get same length!')

        if type(X) != pd.DataFrame:
            raise TypeError('train feature X must be a pandas.DataFrame!')
        else:
            train_feature = X
        num_train_ins = len(X.index)

        if type(y) != pd.Series:
            self.trainLabel = pd.Series(y, index=X.index, name='label')
        else:
            self.trainLabel = y
            if self.trainLabel.name != 'label':
                self.trainLabel.name = 'label'

        # if categorical, encode to one-hot code
        # note threshold rate parameter: oneHotThre
        train_feature.apply(find_cate_feat, args=(self.cateFeatColList, self.cateFeatList))

        self.numFeatColList = train_feature.columns.difference(self.cateFeatColList)
        train_feature_num = train_feature.ix[:, self.numFeatColList]

        # one-Hot code column name:
        self.oneHotCol = ['oneHot_'+str(num) for num in range(len(self.cateFeatList))]

        # dealing with train categorical features:
        cate_feat_ser = pd.Series(self.cateFeatList, index=self.oneHotCol)
        # apply one-hot encode function to each row in train_feature[self.cateFeatColList]
        # train_feature_cate = pd.DataFrame(columns=self.oneHotCol)
        train_feature_cate = train_feature[self.cateFeatColList].apply(one_hot_encode, args=(cate_feat_ser, ), axis=1)
        # concat two DataFrame by same index
        self.trainFeat = pd.concat([train_feature_num, train_feature_cate], axis=1)

        # let's learn via XGBoost
        if self.paraDic.get('bootstrap'):
            # use bootstrap to sample
            # seed()
            for bsRound in range(self.paraDic.get('bootStrapRounds')):
                # boot strap for bootStrapRounds rounds
                sample_num = ceil(float(num_train_ins) * self.paraDic.get('bootStrapRato'))
                if self.paraDic.get('bootStrapEqu'):
                    # positive and negative instance should get same weight when sampling
                    pos_idx = self.trainLabel[self.trainLabel['label'] == 1].index
                    neg_idx = self.trainLabel.index.difference(pos_idx)
                    sample_weight = pd.Series(index=X.index)
                    sample_weight[pos_idx] = 0.5 / float(len(pos_idx))
                    sample_weight[neg_idx] = 0.5 / float(len(neg_idx))

                    # currentPosIdx = pos_idx[
                    #     floor(uniform(size=(1, ceil(len(pos_idx) * self.paraDic.get('bootStrapRato')))) * len(pos_idx)).astype(int).tolist()]
                    # currentNegIdx = neg_idx[
                    #     floor(uniform(size=(1, ceil(len(neg_idx) * self.paraDic.get('bootStrapRato')))) * len(neg_idx)).astype(int).tolist()]
                    # posInstance = self.trainFeat.ix[currentPosIdx, :]
                    # negInstance = self.trainFeat.ix[currentNegIdx, :]
                    # store into XGBoost
                    # current_df = posInstance.append(negInstance)
                    # current_label = [True] * len(currentPosIdx) + [False] * len(currentNegIdx)

                    current_df = train_feature.sample(n=sample_num, weights=sample_weight, replace=True)
                    current_label = self.trainLabel[current_df.index]

                else:
                    # don't care if pos instance equal to neg instance
                    # sampleTrainNum = ceil(float(num_train_ins) * self.paraDic.get('bootStrapRato'))
                    # sampleTrainIdx = self.trainFeat.index[floor(uniform(size=(1, sampleTrainNum)) * num_train_ins).astype(int).tolist()]
                    # current_df = self.trainFeat.ix[sampleTrainIdx]
                    # current_label = self.trainLabel.ix[sampleTrainIdx]

                    current_df = train_feature.sample(n=sample_num, replace=True)
                    current_label = self.trainLabel[current_df.index]
                    
                # learn and dump a Booster per round
                current_train_dm = xgb.DMatrix(current_df, label=current_label)
                current_booster = xgb.train(params=self.xgbTrainPara, dtrain=current_train_dm, **self.xgbSetPara)
                print 'Training:\t bootStrap Round #\t%d\n' % bsRound
                self.bsBoosterList.append(current_booster)
        else:
            # don't bootstrap
            # load train_feature to XGBoost.DMatrix
            train_dmat = xgb.DMatrix(self.trainFeat, label=self.trainLabel)
            current_booster = xgb.train(params=self.xgbTrainPara, dtrain=train_dmat, **self.xgbSetPara)
            print 'NO BOOTSTRAP:\ttrain finished.\n'
            self.bsBoosterList.append(current_booster)

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
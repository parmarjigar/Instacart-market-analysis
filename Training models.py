#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import joblib
import gc
import multiprocessing as mp
import numpy as np
import pandas as pd
import _pickle as cpickle
import pickle
from datetime import datetime
import time
from pandas import HDFStore
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
import xgboost
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from operator import itemgetter
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import check_cv, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, log_loss, auc


# In[2]:


train = pd.read_pickle("data/final_train.pkl")
x_train = train.drop(['order_id'], axis=1)
del train;
gc.collect()
test = pd.read_pickle("data/final_test.pkl")
x_test = test.drop(["order_id"], axis=1)
labels = pd.read_pickle("data/final_labels.pkl")


# In[3]:


x_train.shape, x_test.shape


# In[4]:


class CustomStackingClassifier:
    def __init__(self, estimators, random_state, params, nround, 
                 version, loop=3,
                 valid_size=0.05, stratify=True, verbose=1,
                 early_stopping=60, use_probas=True):
        self.clf = estimators
        self.mod=cpickle
        self.loop = loop
        self.params = params
        self.nround = nround    
        self.version = version
        self.valid_size = valid_size
        self.verbose = verbose
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.models = []


    def split_build_valid(self, train_user, X_train, y_train):
        train_user['is_valid'] = np.random.choice(
            [0,1],
            size=len(train_user),
            p=[1-self.valid_size, self.valid_size])

        valid_n = train_user['is_valid'].sum()
        build_n = (train_user.shape[0] - valid_n)
        
        print('build user:{}, valid user:{}'.format(build_n, valid_n))
        valid_user = train_user[train_user['is_valid']==1].user_id
        is_valid = X_train.user_id.isin(valid_user)
        
        dbuild = lgb.Dataset(X_train[~is_valid].drop('user_id', axis=1),
                             y_train[~is_valid],
                             categorical_feature=['product_id', 'aisle_id', 'department_id'])
        dvalid = lgb.Dataset(X_train[is_valid].drop('user_id', axis=1),
                             label=y_train[is_valid],
                             categorical_feature=['product_id', 'aisle_id', 'department_id'])
        watchlist_set = [dbuild, dvalid]
        watchlist_name = ['build', 'valid']
        
        print('FINAL SHAPE')
        print('dbuild.shape:{}  dvalid.shape:{}\n'.format(
            dbuild.data.shape,
            dvalid.data.shape))
        return dbuild, dvalid, watchlist_set, watchlist_name

    def fit(self, x, y):
        np.random.seed(self.random_state)
        train_user = x[['user_id']].drop_duplicates()

        for i in range(self.loop):
            dbuild, dvalid, watchlist_set, watchlist_name = self.split_build_valid(train_user, x, y)
            gc.collect();

            # Train models
            model = lgb.train(
                self.params,
                dbuild,
                self.nround,
                watchlist_set,
                watchlist_name,
                early_stopping_rounds=self.early_stopping,
                categorical_feature=['product_id', 'aisle_id', 'department_id'],
                verbose_eval=5)
            joblib.dump(model, "lgb_models/lgb_trained_{}_{}".format(self.version, i))
            self.models.append(model)
            del [dbuild, dvalid, watchlist_set, watchlist_name];
            gc.collect();
        del train_user;
        gc.collect()
        return self


    def predict(self, x, test_data):
#         dtest  = lgb.Dataset(x)
        sub_test = test_data[['order_id', 'product_id']]
        sub_test['yhat'] = 0
        for model in self.models:
            sub_test['yhat'] += model.predict(x)
        sub_test['yhat'] /= self.loop
        return sub_test


# In[5]:


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 256,
    'min_sum_hessian_in_leaf':20,
    'max_depth': 12,
    'learning_rate': 0.05,
    'feature_fraction': 0.6,
    # 'bagging_fraction': 0.9,
    # 'bagging_freq': 3,
    'verbose': 1
}

cscf_1 = CustomStackingClassifier(lgb, 71, params, 10000, 1)
cscf_2 = CustomStackingClassifier(lgb, 72, params, 10000, 2)
cscf_3 = CustomStackingClassifier(lgb, 73, params, 10000, 3)


# In[6]:


cscf_1.fit(x_train, labels)
stack1 = cscf_1.predict(x_test, test)
stack1.to_csv("data/lgb_stack1.csv", index=False)


# In[7]:


cscf_2.fit(x_train, labels)
stack1 = cscf_2.predict(x_test, test)
stack1.to_csv("data/lgb_stack2.csv", index=False)


# In[8]:


cscf_3.fit(x_train, labels)
stack1 = cscf_3.predict(x_test, test)
stack1.to_csv("data/lgb_stack3.csv", index=False)


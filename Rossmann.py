#encoding:utf-8

import pandas as pd
import datetime
import csv
import numpy as np
import xgboost as xgb
import itertools
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import cross_validation
# from matplotlib import pylab as plt
plot = True
goal = 'Sales'
myid = 'Id'


def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return 'rmspe', rmspe


def get_data():
    store = pd.read_csv('./store.csv')
    train_org = pd.read_csv('./train.csv', dtype={'StateHoliday': pd.np.string_})
    test_org = pd.read_csv('./test.csv', dtype={'StateHoliday': pd.np.string_})
    train = pd.merge(train_org, store, on='Store', how='left')
    test = pd.merge(test_org, store, on='Store', how='left')

    features1 = ['Id', 'Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday',
                 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance',
                 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

    feature_non = ['Promo', 'Store', 'Date', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'Promo2',
                   'PromoInterval']
    return (train, test, features1, feature_non)


def process_data(train, test, features, features_no):
    train = train[train['Open'] == 1]

    test = test[test['Open'] == 1]
    for rossm in [train, test]:
        rossm['year'] = rossm.Date.apply(lambda x: int(x.split('-')[0]))
        rossm['month'] = rossm.Date.apply(lambda x: int(x.split('-')[1]))
        rossm['day'] = rossm.Date.apply(lambda x: int(x.split('-')[2]))

        rossm['promojan'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jan' in x else 0)
        rossm['promofeb'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Feb' in x else 0)
        rossm['promomar'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Mar' in x else 0)
        rossm['promoapr'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Apr' in x else 0)
        rossm['promomay'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'May' in x else 0)
        rossm['promojun'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jun' in x else 0)
        rossm['promojul'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jul' in x else 0)
        rossm['promoaug'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Aug' in x else 0)
        rossm['promosep'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Sep' in x else 0)
        rossm['promooct'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Oct' in x else 0)
        rossm['promonov'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Nov' in x else 0)
        rossm['promodec'] = rossm.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Dec' in x else 0)

    day_dummies = pd.get_dummies(train['DayOfWeek'], prefix='Day')
    day_dummies.drop(['Day_7'], axis=1, inplace=True)  # 删除周日的数据
    train = train.join(day_dummies)

    day_dummies_test = pd.get_dummies(test['DayOfWeek'], prefix='Day')
    day_dummies_test.drop(['Day_7'], axis=1, inplace=True)  # 删除周日数据
    test = test.join(day_dummies_test)

    no_feature = [myid, 'Date']
    features = set(features) - set(no_feature)
    features_no = set(features_no) - set(no_feature)
    features = list(features)
    features.extend(['year', 'month', 'day'])

    class DataFrameInputer(TransformerMixin):
        def __init__(self):
            """
            """

        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0]
                                   if X[c].dtype == np.dtype('O')
                                   else X[c].mean() for c in X], index=X.columns)
            return self

        def transform(self, X, y=None):
            return X.fillna(self.fill)

    train = DataFrameInputer().fit_transform(train)
    test = DataFrameInputer().fit_transform(test)

    le = LabelEncoder()
    for col in features:
        le.fit(list(train[col]) + list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    scaler = StandardScaler()
    print(set(features) - set(features_no) - set([]))
    for col in set(features) - set(features_no) - set([]):
        try:
            scaler.fit(list(train[col]) + list(test[col]))
        except:
            print(col)
        train[col] = scaler.transform(train[col])
        test[col] = scaler.transform(test[col])
    return (train, test, features, features_no)

train,test,features,features_non_numeric = get_data()
train,test,features,features_non_numeric = process_data(train,test,features,features_non_numeric)
train[['Promo2SinceWeek']].head()
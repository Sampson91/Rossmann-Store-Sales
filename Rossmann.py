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
from matplotlib import pylab as plt
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

def load_data():
    store = pd.read_csv('./store.csv')
    train_org = pd.read_csv('./train.csv',dtype={'StateHoliday':pd.np.string_})
    test_org = pd.read_csv('./test.csv',dtype={'StateHoliday':pd.np.string_})
    train = pd.merge(train_org,store,on='Store',how='left')
    test = pd.merge(test_org,store,on='Store',how='left')
    feature = test.columns.tolist()
    numerics = ['int16','int32','int64','float16','float32','float64']
    feature_numeric = test.select_dtypes(include = numerics).columns.tolist()
    feature_non_numeric = [f for f in feature if f not in feature_numeric]
    return (train,test,feature,feature_non_numeric)


def process_data(train, test, features, features_non_numeric):
    train = train[train['Sales'] > 0]

    for data in [train, test]:
        data['year'] = data.Date.apply(lambda x: x.split('-')[0])
        data['year'] = data['year'].astype(float)
        data['month'] = data.Date.apply(lambda x: x.split('-')[1])
        data['month'] = data['month'].astype(float)
        data['day'] = data.Date.apply(lambda x: x.split('-')[2])
        data['day'] = data['day'].astype(float)

        data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jan' in x else 0)
        data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Feb' in x else 0)
        data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Mar' in x else 0)
        data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Apr' in x else 0)
        data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'May' in x else 0)
        data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jun' in x else 0)
        data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Jul' in x else 0)
        data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Aug' in x else 0)
        data['promosep'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Sep' in x else 0)
        data['promooct'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Oct' in x else 0)
        data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Nov' in x else 0)
        data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if 'Dec' in x else 0)

    noisy_features = [myid, 'Date']
    features = [c for c in features if c not in noisy_features]
    features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]
    features.extend(['year', 'month', 'day'])

    class DataFrameInputer(TransformerMixin):

        def __init__(self):
            """
            """

        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0]
                                   if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                                  index=X.columns)
            return self

        def transform(self, X, y=None):
            return X.fillna(self.fill)

    train = DataFrameInputer().fit_transform(train)
    test = DataFrameInputer().fit_transform(test)

    le = LabelEncoder()
    for col in features_non_numeric:
        le.fit(list(train[col]) + list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    scaler = StandardScaler()
    #     print(set(features) - set(features_non_numeric) - set([]))

    #     print('feature',features)
    #     print('features_non',features_non_numeric)
    #     print(test['StateHoliday'].head())
    for col in set(features) - set(features_non_numeric) - set([]):
        #         print(col)
        try:
            scaler.fit(list(train[col]) + list(test[col]))
        except:
            print(col)
        train[col] = scaler.transform(train[col])
        test[col] = scaler.transform(test[col])
    return (train, test, features, features_non_numeric)

train,test,features,features_non_numeric = load_data()
train,test,features,features_non_numeric = process_data(train,test,features,features_non_numeric)
train[['Promo2SinceWeek']].head()
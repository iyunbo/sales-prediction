import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from preperation import rmspe, rmspe_xg

seed = 42


def run_linear_regression(train_x, train_y, test_x, test_y):
    regressor = LinearRegression()
    regressor.fit(train_x, train_y)
    predict = regressor.predict(test_x)
    print('LinearRegression RMSPE =', rmspe(yhat=predict, y=test_y))


def run_random_forest(train_x, train_y, test_x, test_y):
    regressor = RandomForestRegressor()
    regressor.fit(train_x, train_y)
    predict = regressor.predict(test_x)
    print('RandomForestRegressor RMSPE = ', rmspe(predict, test_y))


def run_xgboost(train_x, train_y, test_x, test_y):
    regressor = xgb.XGBRegressor(max_depth=12,
                                 eta=0.02,
                                 subsample=0.9,
                                 colsample_bytree=0.7,
                                 objective='reg:linear',
                                 nthread=8,
                                 seed=seed)
    regressor.fit(train_x, train_y)
    print('XGBoost RMSPE =', rmspe(regressor.predict(test_x), test_y))
    # dtrain = xgb.DMatrix(train_x, train_y)
    # dtest = xgb.DMatrix(test_x, test_y)
    #
    # num_round = 20000
    # evallist = [(dtrain, 'train'), (dtest, 'test')]
    # param = {'bst:max_depth': 12,
    #          'bst:eta': 0.02,
    #          'subsample': 0.9,
    #          'colsample_bytree': 0.7,
    #          'silent': 1,
    #          'objective': 'reg:linear',
    #          'nthread': 8,
    #          'seed': seed}
    #
    # model = xgb.train(param, dtrain, num_round, evallist, feval=rmspe_xg, verbose_eval=300, early_stopping_rounds=300)
    #
    # dpred = xgb.DMatrix(test_x)
    # ypred = model.predict(dpred)
    # print('XGBoost RMSPE =', rmspe(ypred, test_y))


def train_test(df, features_x, feature_y):
    train = df.loc[(df['DateInt'] <= 20150615)]
    test = df.loc[(df['DateInt'] > 20150615)]
    return train[features_x], test[features_x], train[feature_y], test[feature_y]


def run_models(df_train, features_x, feature_y):
    train_x, test_x, train_y, test_y = train_test(df_train, features_x, feature_y)
    run_linear_regression(train_x, train_y, test_x, test_y)
    run_random_forest(train_x, train_y, test_x, test_y)
    run_xgboost(train_x, train_y, test_x, test_y)

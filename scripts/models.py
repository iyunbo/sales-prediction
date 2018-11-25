import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.scorer import make_scorer
from preperation import rmspe, rmspe_xg, rmspe_score
from sklearn.model_selection import RandomizedSearchCV

seed = 42


def run_linear_regression(train_x, train_y, test_x, test_y):
    regressor = LinearRegression()
    regressor.fit(train_x, train_y)
    predict = regressor.predict(X=test_x)
    print('LinearRegression RMSPE =', rmspe(yhat=predict, y=test_y))


def run_random_forest(train_x, train_y, test_x, test_y, params=None):
    regressor = RandomForestRegressor()
    regressor.set_params(params)
    regressor.fit(train_x, train_y)
    predict = regressor.predict(test_x)
    print('RandomForestRegressor RMSPE = ', rmspe(predict, test_y))


def run_xgboost(train_x, train_y, test_x, test_y):
    regressor = xgb.XGBRegressor(nthread=8)
    regressor.fit(train_x, train_y)
    predict = regressor.predict(test_x)
    print('XGBoost RMSPE =', rmspe(predict, test_y))


def train_test(df, features_x, feature_y):
    train = df.loc[(df['Type'] == 'train')]
    test = df.loc[(df['Type'] == 'validation')]
    return train[features_x], test[features_x], train[feature_y], test[feature_y]


def run_models(df_train, features_x, feature_y):
    train_x, test_x, train_y, test_y = train_test(df_train, features_x, feature_y)
    run_linear_regression(train_x, train_y, test_x, test_y)
    run_random_forest(train_x, train_y, test_x, test_y)
    run_xgboost(train_x, train_y, test_x, test_y)


def get_scorer():
    return make_scorer(rmspe_score, greater_is_better=False)


def train_random_forest(df_train, features_x, feature_y, tuned_params=None):
    # split train test
    train_x, test_x, train_y, test_y = train_test(df_train, features_x, feature_y)
    # init model
    random_forest = RandomForestRegressor(**tuned_params)
    # parameters space
    hyperparameters = dict(
        n_estimators=[10, 50, 100, 200],
        max_features=['auto'],
        max_depth=[10, 20, 50, 100, 200, None],
        min_samples_leaf=[10, 50, 100, 200, 400],
        bootstrap=[True],
        min_samples_split=[2, 5, 10],
        max_leaf_nodes=[None],
        min_impurity_decrease=[0],
        min_weight_fraction_leaf=[0],
        oob_score=[True],
        warm_start=[True]
    )

    if tuned_params:
        # training
        best_model = random_forest.fit(train_x, train_y)
    else:
        # create random search
        regressor = RandomizedSearchCV(random_forest, hyperparameters, random_state=seed, n_iter=100, cv=5,
                                       scoring=get_scorer(), verbose=3, n_jobs=-1)
        # training
        best_model = regressor.fit(train_x, train_y)
    # evaluation
    predict = best_model.predict(test_x)
    print('Improved RandomForestRegressor RMSPE = ', rmspe(predict, test_y))
    if not tuned_params:
        print('Best parameters:', best_model.best_estimator_.get_params())


def train_xgboost(df_train, features_x, feature_y):
    # split train test
    train_x, test_x, train_y, test_y = train_test(df_train, features_x, feature_y)
    # prepare data structure for xgb
    dtrain = xgb.DMatrix(train_x, train_y)
    dtest = xgb.DMatrix(test_x, test_y)
    # setup parameters
    num_round = 3000
    evallist = [(dtrain, 'train'), (dtest, 'test')]
    # training
    params = {'bst:max_depth': 12,
              'bst:eta': 0.01,
              'subsample': 0.9,
              'colsample_bytree': 0.7,
              'objective': 'reg:linear',
              'booster': 'gbtree',
              'nthread': 6,
              'seed': seed,
              'silent': True}

    print(params)
    best_model = xgb.train(params, dtrain, num_round, evallist, feval=rmspe_xg, verbose_eval=100,
                           early_stopping_rounds=500)
    xgb.plot_importance(best_model)
    # evaluation
    predict = best_model.predict(dtest)
    print('Improved XGBoost RMSPE = ', rmspe(predict, test_y))

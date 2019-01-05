import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.scorer import make_scorer
from preperation import rmspe, rmspe_xg, rmspe_score
from sklearn.model_selection import RandomizedSearchCV
import time

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


def tune_xgboost(df_train, features_x, feature_y):
    start_time = time.time()
    # split train test
    train_x, test_x, train_y, test_y = train_test(df_train, features_x, feature_y)
    param_grid = {
        'tree_method': ['gpu_hist'],
        'silent': [False],
        'max_depth': [12, 13, 14, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0, 3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100]}

    regressor = xgb.XGBRegressor(nthread=2)

    rs_regressor = RandomizedSearchCV(regressor,
                                      param_grid,
                                      n_iter=50,
                                      n_jobs=4,
                                      verbose=1,
                                      cv=5,
                                      scoring=get_scorer(),
                                      refit=False,
                                      random_state=seed)
    rs_regressor.fit(train_x, train_y, eval_metric=rmspe_xg, early_stopping_rounds=30, eval_set=[(test_x, test_y)])

    best_score = rs_regressor.best_score_
    best_params = rs_regressor.best_params_
    print("Best score: {}".format(best_score))
    print("Best params: ")
    for param_name in sorted(best_params.keys()):
        print('%s: %r' % (param_name, best_params[param_name]))

    best_model = xgb.XGBRegressor(nthread=8, params=rs_regressor.best_params_)
    best_model.fit(train_x, train_y, eval_metric=rmspe_xg, early_stopping_rounds=30, eval_set=[(test_x, test_y)])
    predict = best_model.predict(test_x)
    print('Improved XGBoost RMSPE = ', rmspe(predict, test_y))

    print("--- %s hours ---" % ((time.time() - start_time) / (60 * 60)))

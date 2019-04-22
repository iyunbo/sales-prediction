import datetime as dt
import json
import logging as log
import os.path as path
import subprocess
import time
from io import StringIO

import pandas as pd
import xgboost as xgb
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sqlalchemy import create_engine

from .preparation import local_data_dir
from .preparation import rmspe, rmspe_xg, rmspe_score

seed = 16

xgb_params = {'max_depth': 12,
              'learning_rates': 0.1,
              'gamma': 0.5,
              'colsample_bytree': 0.6,
              'colsample_bylevel': 0.5,
              'min_child_weight': 5.0,
              'n_estimator': 140,
              'reg_lambda': 100.0,
              'subsample': 0.6,
              'nthread': 7,
              'random_state': seed,
              'tree_method': 'gpu_hist',
              'silent': True}

submission_file = 'submission.csv'


def run_linear_regression(train_x, train_y, validation_x, validation_y):
    regressor = LinearRegression(n_jobs=6)
    regressor.fit(train_x, train_y)
    predict = regressor.predict(validation_x)
    return rmspe(yhat=predict, y=validation_y)


def run_random_forest(train_x, train_y, validation_x, validation_y):
    regressor = RandomForestRegressor(n_jobs=6, random_state=seed)
    regressor.fit(train_x, train_y)
    predict = regressor.predict(validation_x)
    return rmspe(predict, validation_y)


def run_xgboost(train_x, train_y, validation_x, validation_y):
    regressor = xgb.XGBRegressor(nthread=6, random_state=seed)
    regressor.fit(train_x, train_y)
    predict = regressor.predict(validation_x)
    return rmspe(predict, validation_y)


def train_validation(df, features_x, feature_y):
    train = df.loc[(df['Type'] == 'train')]
    validation = df.loc[(df['Type'] == 'validation')]
    return train[features_x], validation[features_x], train[feature_y], validation[feature_y]


def run_models(df_train, features_x, feature_y):
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    run_linear_regression(train_x, train_y, validation_x, validation_y)
    run_random_forest(train_x, train_y, validation_x, validation_y)
    run_xgboost(train_x, train_y, validation_x, validation_y)


def quick_score(df_train, features_x, feature_y):
    start_time = time.time()
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    score1 = run_linear_regression(train_x, train_y, validation_x, validation_y)
    log.info("linear regression score = {}".format(score1))
    score2 = run_xgboost(train_x, train_y, validation_x, validation_y)
    log.info("xgboost score = {}".format(score2))
    score3 = run_random_forest(train_x, train_y, validation_x, validation_y)
    log.info("random forest score = {}".format(score3))
    return score1, score2, score3, (time.time() - start_time)


def run_trained_models(df_train, features_x, feature_y):
    # run trained random forest
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    model_file = "random-forest.joblib"
    print("loading model from [", model_file, "]")
    regressor = load(path.join(local_data_dir, model_file))
    predict = regressor.predict(validation_x)
    print("RandomForest RMSPE =", rmspe(predict, validation_y))

    # run trained xgboost
    bst = xgb.Booster({"nthread": 8})
    model_file = "xgboost.model"
    print("loading xgboost from [", model_file, "]")
    bst.load_model("../data/" + model_file)
    dvalidation = xgb.DMatrix(validation_x, validation_y)
    predict = bst.predict(dvalidation, ntree_limit=4053)
    score = rmspe_xg(predict, dvalidation)
    print("XGBoost RMSPE =", score[1])


def final_model():
    final_regressor = xgb.Booster({'nthread': 8})  # init model
    final_regressor.load_model(path.join(local_data_dir, 'xgboost.model'))  # load data

    return final_regressor


def get_scorer():
    return make_scorer(rmspe_score, greater_is_better=False)


def tune_random_forest(df_train, features_x, feature_y):
    start_time = time.time()
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    # init model
    random_forest = RandomForestRegressor()
    # parameters space
    hyperparameters = dict(
        n_estimators=[10, 50, 100, 150],
        max_features=['auto'],
        max_depth=[10, 20, 50, 100, 200, None],
        min_samples_leaf=[2, 5, 10, 50, 100],
        bootstrap=[True],
        min_samples_split=[2, 3, 4, 5, 7, 10],
        max_leaf_nodes=[None],
        min_impurity_decrease=[0],
        min_weight_fraction_leaf=[0],
        oob_score=[True],
        warm_start=[True]
    )

    # create random search
    random_search = RandomizedSearchCV(random_forest, hyperparameters, random_state=seed, n_iter=50, cv=5,
                                       scoring=get_scorer(),
                                       verbose=3, n_jobs=6)
    # training
    random_search.fit(train_x, train_y)
    print_tuning_result(random_search)
    print("--- %.2f hours ---" % ((time.time() - start_time) / (60 * 60)))


def train_random_forest(df_train, features_x, feature_y):
    start_time = time.time()
    tuned_params = {'bootstrap': True,
                    'criterion': 'mse',
                    'max_depth': 200,
                    'max_features': 'auto',
                    'max_leaf_nodes': None,
                    'min_impurity_decrease': 0,
                    'min_impurity_split': None,
                    'min_samples_leaf': 10,
                    'min_samples_split': 2,
                    'min_weight_fraction_leaf': 0,
                    'n_estimators': 150,
                    'n_jobs': 6,
                    'oob_score': True,
                    'random_state': seed,
                    'verbose': 0,
                    'warm_start': True
                    }
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    random_forest = RandomForestRegressor(**tuned_params)
    # training
    regressor = random_forest.fit(train_x, train_y)
    evaluate_model(regressor, 'random-forest', validation_x, validation_y)
    print("--- %.2f hours ---" % ((time.time() - start_time) / (60 * 60)))


def evaluate_model(best_model, title, validation_x, validation_y):
    # evaluation
    predict = best_model.predict(validation_x)
    score = rmspe(predict, validation_y)
    print('Improved ' + title + ' RMSPE = ', score)
    if score < 0.15:
        dump(best_model, 'data/' + title + '.joblib')

    # time consuming
    # plot_learning_curve(best_model, title + " Learning Curve", train_x, train_y,
    #                     n_jobs=6)


def train_xgboost(df_train, features_x, feature_y, num_round=2000, early_stopping_rounds=200):
    start_time = time.time()
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    # prepare data structure for xgb
    dtrain = xgb.DMatrix(train_x, train_y)
    dvalidation = xgb.DMatrix(validation_x, validation_y)
    # setup parameters
    evallist = [(dtrain, 'train'), (dvalidation, 'validation')]

    print(xgb_params)
    best_model = xgb.train(xgb_params, dtrain, num_round, evallist, feval=rmspe_xg, verbose_eval=40,
                           early_stopping_rounds=early_stopping_rounds)
    predict = best_model.predict(dvalidation, ntree_limit=best_model.best_ntree_limit)
    score = rmspe_xg(predict, dvalidation)
    print('best tree limit:', best_model.best_ntree_limit)
    print('XGBoost RMSPE = ', score)
    xgb.plot_importance(best_model)
    best_model.save_model(path.join(local_data_dir, 'xgboost.model'))
    duration = (time.time() - start_time) / (60 * 60)
    print("--- %.2f hours ---" % duration)
    return score[1], duration


def train_ensemble(df_train, features_x, feature_y):
    start_time = time.time()
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    # prepare data structure for xgb
    dtrain = xgb.DMatrix(train_x, train_y)
    dvalidation = xgb.DMatrix(validation_x, validation_y)
    engine = create_engine('sqlite:///{}'.format(path.join(local_data_dir, 'model.db')))

    rows_list = []
    for rnd in range(466, 516):
        best_model, score, train_score, ntree_limit, max_round, params = modeling_xgboost(dtrain, dvalidation,
                                                                                          random_state=rnd)
        best_model.save_model(path.join(local_data_dir, "{}-xgboost-{:.5f}.model".format(rnd, score)))
        rows_list.append({
            'timestamp': dt.datetime.now(),
            'random_state': rnd,
            'score': score,
            'train_score': train_score,
            'model': 'xgboost',
            'parameters': json.dumps(params),
            'ntree_limit': ntree_limit,
            'max_round': max_round
        })
        print('best tree limit:', best_model.best_ntree_limit)
        print('score = ', score)
        print('train score = ', train_score)

    models = pd.DataFrame(rows_list)
    models.to_sql('model_%s' % dt.datetime.now().strftime('%y%m%d%H%M%S'), engine)
    print("--- %.2f hours ---" % ((time.time() - start_time) / (60 * 60)))


def modeling_xgboost(train, validation, random_state=seed):
    # setup parameters
    num_round = 2000
    evallist = [(train, 'train'), (validation, 'validation')]
    xgb_params['random_state'] = random_state
    # training
    print(xgb_params)
    best_model = xgb.train(xgb_params, train, num_round, evallist, feval=rmspe_xg, verbose_eval=100,
                           early_stopping_rounds=200)
    predict = best_model.predict(validation, ntree_limit=best_model.best_ntree_limit)
    score = rmspe_xg(predict, validation)[1]
    predict = best_model.predict(train, ntree_limit=best_model.best_ntree_limit)
    train_score = rmspe_xg(predict, train)[1]

    return best_model, score, train_score, best_model.best_ntree_limit, num_round, xgb_params


# --- 1.78 hours ---
def tune_xgboost(df_train, features_x, feature_y):
    start_time = time.time()
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    param_grid = {
        'tree_method': ['gpu_hist'],
        'silent': [False],
        'max_depth': [12],
        'learning_rate': [0.01, 0.02, 0.1],
        'subsample': [0.6],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0.5, 0.52, 0.54, 0.56],
        'reg_lambda': [1.0, 5.0, 10.0, 50.0, 100.0, 120, 130],
        'n_estimators': [100, 110, 120, 130]}

    regressor = xgb.XGBRegressor(nthread=-1)

    random_search = RandomizedSearchCV(regressor,
                                       param_grid,
                                       n_iter=80,
                                       n_jobs=6,
                                       verbose=1,
                                       cv=5,
                                       scoring=get_scorer(),
                                       refit=False,
                                       random_state=seed)
    random_search.fit(train_x, train_y, eval_metric=rmspe_xg, early_stopping_rounds=50,
                      eval_set=[(validation_x, validation_y)])

    print_tuning_result(random_search)
    print("--- %.2f hours ---" % ((time.time() - start_time) / (60 * 60)))


def print_tuning_result(random_search):
    best_score = random_search.best_score_
    best_params = random_search.best_params_
    print("Best score: {}".format(best_score))
    print("Best params: ")
    for param_name in sorted(best_params.keys()):
        print('%s: %r' % (param_name, best_params[param_name]))


def summit(message):
    log.info('submitting {} ...'.format(message))
    result = subprocess.check_output(
        ['kaggle', 'competitions', 'submit', '-c', 'rossmann-store-sales', '-f',
         path.join(local_data_dir, 'submission.csv'), '-q', '-m', message])
    print(result.decode("utf-8"))


def get_kaggle_score(message):
    log.info('getting result of {} ...'.format(message))
    time.sleep(5)
    result = subprocess.check_output(
        ['kaggle', 'competitions', 'submissions', '-c', 'rossmann-store-sales', '-v'])
    csv = StringIO(result.decode("utf-8"))
    df = pd.read_csv(csv)
    result = df[df['description'] == message].iloc[0]
    return result


def save_result(test, prediction):
    test['Id'] = test.Id.astype(int)
    test = test.set_index('Id')
    test['Sales'] = prediction
    result = test[['Sales']].copy()
    log.info("saving result:")
    print(result.sort_index().head())
    result.to_csv(path.join(local_data_dir, submission_file))

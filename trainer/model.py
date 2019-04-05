import time

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve

from .preparation import rmspe, rmspe_xg, rmspe_score

seed = 16


def run_linear_regression(train_x, train_y, validation_x, validation_y):
    regressor = LinearRegression(n_jobs=6)
    regressor.fit(train_x, train_y)
    predict = regressor.predict(validation_x)
    print('LinearRegression RMSPE =', rmspe(yhat=predict, y=validation_y))


def run_random_forest(train_x, train_y, validation_x, validation_y):
    regressor = RandomForestRegressor(n_jobs=32, random_state=seed)
    regressor.fit(train_x, train_y)
    predict = regressor.predict(validation_x)
    score = rmspe(predict, validation_y)
    return score


def run_xgboost(train_x, train_y, validation_x, validation_y):
    regressor = xgb.XGBRegressor(nthread=6, random_state=seed)
    regressor.fit(train_x, train_y)
    predict = regressor.predict(validation_x)
    print('XGBoost RMSPE =', rmspe(predict, validation_y))


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
    score = run_random_forest(train_x, train_y, validation_x, validation_y)
    return score, (time.time() - start_time)


def run_trained_models(df_train, features_x, feature_y):
    # run trained random forest
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    model_file = "random-forest.joblib"
    print("loading model from [", model_file, "]")
    regressor = load("../data/" + model_file)
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


def final_model(df_train, features_x, feature_y):
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
    random_forest = RandomForestRegressor(**tuned_params)
    # training
    final_regressor = random_forest.fit(df_train[features_x], df_train[feature_y])
    print("--- %.2f hours ---" % ((time.time() - start_time) / (60 * 60)))

    # start_time = time.time()
    # train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    # # prepare data structure for xgb
    # dtrain = xgb.DMatrix(train_x, train_y)
    # dvalidation = xgb.DMatrix(validation_x, validation_y)
    # # setup parameters
    # num_round = 8000
    # evallist = [(dtrain, 'train'), (dvalidation, 'validation')]
    # # training
    # params = {'bst:max_depth': 12,
    #           'bst:eta': 0.01,
    #           'gamma': 1.0,
    #           'colsample_bytree': 1.0,
    #           'colsample_bylevel': 0.6,
    #           'min_child_weight': 5.0,
    #           'n_estimator': 80,
    #           'reg_lambda': 10.0,
    #           'subsample': 1.0,
    #           'nthread': 6,
    #           'seed': seed,
    #           'tree_method': 'gpu_hist',
    #           'silent': True}
    #
    # print(params)
    # final_regressor = xgb.train(params, dtrain, num_round, evallist, feval=rmspe_xg, verbose_eval=100,
    #                             early_stopping_rounds=600)
    # predict = final_regressor.predict(dvalidation, ntree_limit=final_regressor.best_ntree_limit)
    # score = rmspe_xg(predict, dvalidation)
    # print('best tree limit:', final_regressor.best_ntree_limit)
    # print('XGBoost RMSPE = ', score)
    # print("--- %.2f hours ---" % ((time.time() - start_time) / (60 * 60)))

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
        dump(best_model, '../data/' + title + '.joblib')

    # time consuming
    # plot_learning_curve(best_model, title + " Learning Curve", train_x, train_y,
    #                     n_jobs=6)


def train_xgboost(df_train, features_x, feature_y):
    start_time = time.time()
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    # prepare data structure for xgb
    dtrain = xgb.DMatrix(train_x, train_y)
    dvalidation = xgb.DMatrix(validation_x, validation_y)
    # setup parameters
    num_round = 8000
    evallist = [(dtrain, 'train'), (dvalidation, 'validation')]
    # training
    params = {'bst:max_depth': 12,
              'bst:eta': 0.01,
              'gamma': 1.0,
              'colsample_bytree': 1.0,
              'colsample_bylevel': 0.6,
              'min_child_weight': 5.0,
              'n_estimator': 80,
              'reg_lambda': 10.0,
              'subsample': 1.0,
              'nthread': 6,
              'seed': seed,
              # 'tree_method': 'gpu_hist',
              'silent': True}

    print(params)
    best_model = xgb.train(params, dtrain, num_round, evallist, feval=rmspe_xg, verbose_eval=100,
                           early_stopping_rounds=600)
    predict = best_model.predict(dvalidation, ntree_limit=best_model.best_ntree_limit)
    score = rmspe_xg(predict, dvalidation)
    print('best tree limit:', best_model.best_ntree_limit)
    print('XGBoost RMSPE = ', score)
    xgb.plot_importance(best_model)
    best_model.save_model('../data/xgboost.model')
    print("--- %.2f hours ---" % ((time.time() - start_time) / (60 * 60)))


# --- 1.78 hours ---
def tune_xgboost(df_train, features_x, feature_y):
    start_time = time.time()
    train_x, validation_x, train_y, validation_y = train_validation(df_train, features_x, feature_y)
    param_grid = {
        # 'tree_method': ['gpu_hist'],
        'silent': [False],
        'max_depth': [12, 13, 14, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0, 3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [50, 80, 100]}

    regressor = xgb.XGBRegressor(nthread=2)

    random_search = RandomizedSearchCV(regressor,
                                       param_grid,
                                       n_iter=20,
                                       n_jobs=32,
                                       verbose=1,
                                       cv=5,
                                       scoring=get_scorer(),
                                       refit=False,
                                       random_state=seed)
    random_search.fit(train_x, train_y, eval_metric=rmspe_xg, early_stopping_rounds=30,
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


def plot_learning_curve(estimator, title, x_train, y_train, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    x_train : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_train : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, x_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(validation_scores, axis=1)
    val_scores_std = np.std(validation_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

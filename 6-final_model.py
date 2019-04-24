import os.path as path

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine

from trainer.model import save_result, summit, get_kaggle_score
from trainer.preparation import load_data, extract_features, local_data_dir, log

NTREE_LIMIT = 570
ENSEMBLE = True
TOP = 20
TRIALS = [1, 3, 5, 10, 20, 30, 40, 50]
MODEL_SUITE = 'model_6'


def sub_msg(top):
    model_type = "ensemble" if ENSEMBLE else "xgboost"
    top = top if ENSEMBLE else NTREE_LIMIT
    return "{}-{}-top-{}".format(MODEL_SUITE, model_type, top)


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    df_test = feat_matrix.loc[feat_matrix['Type'] == 'test'].copy()
    results = []
    for top in TRIALS:
        predict = forecast(df_test, features_x, ENSEMBLE, top)
        save_result(df_test, predict)
        summit(sub_msg(top))
        result = get_kaggle_score(sub_msg(top))
        results.append((top, result))

    print(results)


def forecast(df_test, features_x, ensemble=True, top=TOP):
    if ensemble:
        prediction = ensemble_forecast(df_test, features_x, top)
    else:
        prediction = simple_forecast(df_test, features_x)

    return np.expm1(prediction).astype(int)


def ensemble_forecast(df_test, features_x, top):
    engine = create_engine('sqlite:///{}'.format(path.join(local_data_dir, 'model.db')))
    selected_index = pd.read_sql_table('{}_score'.format(MODEL_SUITE), engine) \
        .sort_values(by='private_score').head(top).index
    model = pd.read_sql_table(MODEL_SUITE, engine, parse_dates=['timestamp']).iloc[selected_index]
    prediction = ensemble_predict(df_test, features_x, model)
    return prediction


def simple_forecast(df_test, features_x, ntree_limit=NTREE_LIMIT):
    file = path.join(local_data_dir, "xgboost.model")
    model = xgb.Booster({'nthread': 8})  # init model
    model.load_model(file)  # load data
    predict = model.predict(xgb.DMatrix(df_test[features_x]), ntree_limit=ntree_limit)
    return predict


def ensemble_predict(df_test, features_x, models):
    predictions = []
    weights = []
    log.info("models size: {}".format(models.shape))
    for rnd in models['random_state']:
        score = models[models['random_state'] == rnd]['score'].iloc[0]
        ntree_limit = models[models['random_state'] == rnd]['ntree_limit'].iloc[0]
        file = path.join(local_data_dir, "{}-xgboost-{:.5f}.model".format(rnd, score))
        model = xgb.Booster({'nthread': 8})  # init model
        model.load_model(file)  # load data
        predict = model.predict(xgb.DMatrix(df_test[features_x]), ntree_limit=ntree_limit)
        predictions.append(predict)
        weights.append((1 - score) / ntree_limit)
    length = df_test.shape[0]
    prediction = np.ndarray(shape=length, )
    for ind in range(length):
        values = []
        for predict in predictions:
            values.append(predict[ind])
        prediction[ind] = np.average(values, weights=weights)
    return prediction


if __name__ == '__main__':
    main()

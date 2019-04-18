import os.path as path

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine

from trainer.preparation import load_data, extract_features, local_data_dir, log


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    df_test = feat_matrix.loc[feat_matrix['Type'] == 'test'].copy()
    predict = forecast(df_test, features_x)
    df_test['Id'] = df_test.Id.astype(int)
    df_test = df_test.set_index('Id')
    df_test['Sales'] = predict
    result = df_test[['Sales']].copy()
    print(result.head())
    result.to_csv(path.join(local_data_dir, 'submission.csv'))


def forecast(df_test, features_x):
    engine = create_engine('sqlite:///{}'.format(path.join(local_data_dir, 'model.db')))
    list_mod = []
    top = 10
    for table in ['model_2']:
        mod = pd.read_sql_table(table, engine, parse_dates=['timestamp']).sort_values(by='score').head(top)
        list_mod.append(mod)

    # models = pd.concat(list_mod, ignore_index=True)
    # prediction = ensemble_predict(df_test, features_x, models)
    prediction = simple_predict(df_test, features_x)

    return np.expm1(prediction).astype(int)


def simple_predict(df_test, features_x, ntree_limit=614):
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

import os.path as path

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine

from trainer.model import summit, save_result, get_kaggle_score
from trainer.preparation import load_data, extract_features, local_data_dir

MODEL_SUITE = 'model_4'


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    df_test = feat_matrix.loc[feat_matrix['Type'] == 'test'].copy()
    forecast(df_test, features_x)


def forecast(df_test, features_x):
    engine = create_engine('sqlite:///{}'.format(path.join(local_data_dir, 'model.db')))
    results = []
    models = pd.read_sql_table(MODEL_SUITE, engine, parse_dates=['timestamp'])
    for index, mod in models.iterrows():
        test = df_test.copy()
        prediction = predict(test, features_x, mod)
        save_result(test, prediction)
        message = 'xgboost-{}-{}'.format(mod['index'], mod['random_state'])
        summit(message)
        result = get_kaggle_score(message)
        results.append({
            'index': mod['index'],
            'random_state': mod['random_state'],
            'public_score': result['publicScore'],
            'private_score': result['privateScore'],
        })

    res_df = pd.DataFrame(results)
    res_df.to_sql('{}_score'.format(MODEL_SUITE), engine)


def predict(df_test, features_x, model):
    rnd = model['random_state']
    score = model['score']
    ntree_limit = model['ntree_limit']
    file = path.join(local_data_dir, "{}-xgboost-{:.5f}.model".format(rnd, score))
    model = xgb.Booster({'nthread': 8})  # init model
    model.load_model(file)  # load data
    result = model.predict(xgb.DMatrix(df_test[features_x]), ntree_limit=ntree_limit)
    return np.expm1(result).astype(int)


if __name__ == '__main__':
    main()

import os.path as path
import subprocess
from io import StringIO

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine

from trainer.preparation import load_data, extract_features, local_data_dir, log


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    df_test = feat_matrix.loc[feat_matrix['Type'] == 'test'].copy()
    forecast(df_test, features_x)


def summit(index, rnd):
    log.info('submitting ...')
    result = subprocess.check_output(
        ['kaggle', 'competitions', 'submit', '-c', 'rossmann-store-sales', '-f',
         path.join(local_data_dir, 'submission.csv'), '-q', '-m', 'xgboost-{}-{}'.format(index, rnd)])
    log.info('submitted: {}'.format(result.decode("utf-8")))


def get_kaggle_score(index, rnd):
    log.info('getting result ...')
    result = subprocess.check_output(
        ['kaggle', 'competitions', 'submissions', '-c', 'rossmann-store-sales', '-v'])
    csv = StringIO(result.decode("utf-8"))
    df = pd.read_csv(csv)
    log.info('result: {}'.format(df.head(1)))
    res = df[df['description'] == 'xgboost-{}-{}'.format(index, rnd)].iloc[0]
    return res['publicScore'], res['privateScore']


def forecast(df_test, features_x):
    engine = create_engine('sqlite:///{}'.format(path.join(local_data_dir, 'model.db')))
    results = []
    model_suite = 'model_3'
    for table in [model_suite]:
        models = pd.read_sql_table(table, engine, parse_dates=['timestamp'])
        for index, mod in models.iterrows():
            test = df_test.copy()
            prediction = predire(test, features_x, mod)
            save_result(test, prediction)
            summit(mod['index'], mod['random_state'])
            public_score, private_score = get_kaggle_score(mod['index'], mod['random_state'])
            results.append({
                'index': mod['index'],
                'random_state': mod['random_state'],
                'public_score': public_score,
                'private_score': private_score,
            })

    res_df = pd.DataFrame(results)
    res_df.to_sql('{}_score'.format(model_suite), engine)


def save_result(test, prediction):
    test['Id'] = test.Id.astype(int)
    test = test.set_index('Id')
    test['Sales'] = prediction
    result = test[['Sales']].copy()
    print(result.head())
    result.to_csv(path.join(local_data_dir, 'submission.csv'))


def predire(df_test, features_x, model):
    rnd = model['random_state']
    score = model['score']
    ntree_limit = model['ntree_limit']
    file = path.join(local_data_dir, "{}-xgboost-{:.5f}.model".format(rnd, score))
    model = xgb.Booster({'nthread': 8})  # init model
    model.load_model(file)  # load data
    predict = model.predict(xgb.DMatrix(df_test[features_x]), ntree_limit=ntree_limit)
    return np.expm1(predict).astype(int)


if __name__ == '__main__':
    main()

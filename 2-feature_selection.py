import datetime as dt
import os.path as path

import pandas as pd
from sqlalchemy import create_engine

from trainer.model import train_xgboost
from trainer.preparation import load_data, extract_features, log, local_data_dir


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    train_df = feat_matrix.loc[~(feat_matrix['Type'] == 'test')]

    feature_candidates = features_x.copy()

    engine = create_engine('sqlite:///{}'.format(path.join(local_data_dir, 'model.db')))
    rows_list = []

    base_line, duration = train_xgboost(train_df, features_x, feature_y)
    log.info("base line: {:.5f}".format(base_line))

    for feat in features_x:
        feature_candidates.remove(feat)
        score, duration = train_xgboost(train_df, feature_candidates, feature_y)
        result = {
            'timestamp': dt.datetime.now(),
            'feature': feat,
            'score': score,
            'duration': duration,
            'base_line': base_line,
            'delta': base_line - score
        }
        rows_list.append(result)
        feature_candidates.append(feat)
        log.info("{} : {} ".format(feat, score))

    features = pd.DataFrame(rows_list)
    features.to_sql('feature', engine)


if __name__ == '__main__':
    main()

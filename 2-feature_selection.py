import datetime as dt
import os.path as path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from trainer.model import quick_score
from trainer.preparation import load_data, extract_features, log, local_data_dir


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    test_df = feat_matrix.loc[~(feat_matrix['Type'] == 'test')]

    feature_candidates = features_x.copy()

    engine = create_engine('sqlite:///{}'.format(path.join(local_data_dir, 'model.db')))
    rows_list = []

    score1, score2, score3, _ = quick_score(test_df, features_x, feature_y)
    base_line = np.mean([score1, score2, score3])
    log.info("base line: {:.5f}".format(base_line))

    for feat in features_x:
        feature_candidates.remove(feat)
        score_lr, score_xgb, score_rf, duration = quick_score(test_df, feature_candidates, feature_y)
        result = {
            'timestamp': dt.datetime.now(),
            'feature': feat,
            'linear_regression': score_lr,
            'xgboost': score_xgb,
            'random_forest': score_rf,
            'duration': duration,
            'base_line': base_line,
            'delta': base_line - np.mean([score_lr, score_xgb, score_rf])
        }
        rows_list.append(result)
        feature_candidates.append(feat)
        log.info("{} : {} - {} - {}".format(feat, score_lr, score_xgb, score_rf))

    features = pd.DataFrame(rows_list)
    features.to_sql('feature', engine)


if __name__ == '__main__':
    main()

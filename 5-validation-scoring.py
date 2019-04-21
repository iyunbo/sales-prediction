import numpy as np

from trainer.model import train_xgboost
from trainer.preparation import load_data, extract_features, log

IT_COUNT = 5


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    train = feat_matrix.loc[~(feat_matrix['Type'] == 'test')]
    result = []
    for ind in range(IT_COUNT):
        score, duration = train_xgboost(train, features_x, feature_y, num_round=100, early_stopping_rounds=40)
        log.info('score = %.4f' % score)
        log.info('duration : %.2f seconds' % duration)
        result.append(score)

    log.info("result: {}".format(np.mean(result)))


if __name__ == '__main__':
    main()

import os.path as path

import numpy as np

from trainer.model import train_xgboost
from trainer.preparation import load_data, extract_features, log, folk_join, local_data_dir, features_to_file, \
    feat_matrix_pkl

IT_COUNT = 5


def calculate_avg(df):
    df['GroupSalesLog'] = df['SalesLog'].median()
    df['GroupCustomers'] = df['Customers'].median()
    return df


def group_key(row):
    return str(row['Store']) + '-' + str(row['DayOfWeek']) + '-' + str(row['Promo']) + '-' + str(row['SchoolHoliday'])


def improve(df, features):
    log.info("improving ...")
    key_name = 'store-day-promo-holiday'
    df[key_name] = df.apply(group_key, axis=1)
    new_df = folk_join(df, calculate_avg, n_jobs=16, by_feat=key_name)
    new_features = features.copy()
    new_features.append('GroupSalesLog')
    new_features.append('GroupCustomers')
    new_df.to_pickle(path.join(local_data_dir, feat_matrix_pkl))
    features_to_file(new_features)
    log.info("improved: {}".format(new_df.shape))
    return new_df, new_features


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    new_df, new_features = improve(feat_matrix, features_x)
    train = new_df.loc[~(new_df['Type'] == 'test')]
    result = []
    for ind in range(IT_COUNT):
        score, duration = train_xgboost(train, new_features, feature_y, num_round=100, early_stopping_rounds=40)
        log.info('score = %.4f' % score)
        log.info('duration : %.2f seconds' % duration)
        result.append(score)

    log.info("result: {}".format(np.mean(result)))


if __name__ == '__main__':
    main()

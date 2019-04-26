import os.path as path

import numpy as np
import pandas as pd

from trainer.model import train_xgboost
from trainer.preparation import load_data, extract_features, log, local_data_dir, features_to_file, feat_matrix_pkl, \
    folk_join

IT_COUNT = 1
store_states_file = 'store_states.csv'


def calculate_metrics(df):
    total_sales = df['Sales'].sum()
    total_customers = df['Customers'].sum()
    total_sales_holiday = df[df['SchoolHoliday'] == 1]['Sales'].sum()
    total_sales_promo = df[df['Promo'] == 1]['Sales'].sum()
    total_sales_saturday = df[df['DayOfWeek'] == 6]['Sales'].sum()
    df['SalesPerCustomer'] = total_sales / total_customers
    df['RatioOnSchoolHoliday'] = total_sales_holiday / total_sales
    df['RatioOnPromo'] = total_sales_promo / total_sales
    df['RatioOnSaturday'] = total_sales_saturday / total_sales
    return df


def group_key(row):
    return str(row['Store']) + '-' + str(row['DayOfWeek']) + '-' + str(row['Promo']) + '-' + str(row['SchoolHoliday'])


def improve(df, features):
    log.info("improving ...")
    new_df, new_features = calculate_store_metrics(df, features)
    log.info("size of new_df: {}".format(new_df.shape))
    log.info("new features: {}".format(new_features))

    features_to_file(new_features)
    new_df.to_pickle(path.join(local_data_dir, feat_matrix_pkl))

    return new_df, new_features


def calculate_store_metrics(df, features):
    new_features = features
    new_df = folk_join(df, calculate_metrics)
    new_features.append('SalesPerCustomer')
    new_features.append('RatioOnSchoolHoliday')
    new_features.append('RatioOnPromo')
    new_features.append('RatioOnSaturday')
    return new_df, new_features


def integrate_weather(df, features):
    store_states = pd.read_csv(path.join(local_data_dir, 'external', store_states_file))
    weather = []
    new_features = features.copy()
    for state in store_states['State'].unique():
        weather_state = pd.read_csv(path.join(local_data_dir, 'external', 'weather', '{}.csv'.format(state)),
                                    delimiter=';', parse_dates=['Date'])
        weather_state['State'] = state
        weather.append(weather_state[['State', 'Max_TemperatureC', 'Precipitationmm', 'Date']])
    weather_df = pd.concat(weather)
    store_weather = pd.merge(store_states, weather_df, how='left', on=['State'])
    new_df = pd.merge(df, store_weather, how='left', on=['Store', 'Date'])
    return new_df, new_features


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    new_df, new_features = improve(feat_matrix, features_x)
    train = new_df.loc[~(new_df['Type'] == 'test')]
    result = []
    for ind in range(IT_COUNT):
        score, duration = train_xgboost(train, new_features, feature_y)
        log.info('score = %.4f' % score)
        log.info('duration : %.2f seconds' % duration)
        result.append(score)

    log.info("result: {}".format(np.mean(result)))


if __name__ == '__main__':
    main()

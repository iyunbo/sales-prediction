import logging as log
import os
import os.path as path
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

log.basicConfig(level=log.INFO, format='%(asctime)s - [%(name)s] - [%(levelname)s]: %(message)s', stream=sys.stdout)
seed = 42
local_data_dir = 'data'
cloud_data_dir = 'gs://sales-prediction-iyunbo-mlengine/data'
train_filename = 'train.csv'
test_filename = 'test.csv'
store_filename = 'store_comp.csv'
feat_matrix_pkl = 'feat_matrix.pkl'
feat_file = 'features_x.txt'


def load_data(debug=False):
    if already_extracted():
        log.info("features are previously extracted")
        log.info("features: {}".format(read_features()))
        return None, None

    lines = 100 if debug else None
    df_train = pd.read_csv(path.join(local_data_dir, train_filename),
                           parse_dates=['Date'],
                           date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                           nrows=lines,
                           low_memory=False)

    df_test = pd.read_csv(path.join(local_data_dir, test_filename),
                          parse_dates=['Date'],
                          date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                          nrows=lines,
                          low_memory=False)

    df_store = pd.read_csv(path.join(local_data_dir, store_filename), nrows=lines)

    df_train['Type'] = 'train'
    df_test['Type'] = 'test'

    # Combine train and test set
    frames = [df_train, df_test]
    df = pd.concat(frames, sort=True)

    log.info("data loaded: {}".format(df.shape))
    log.info("store loaded: {}".format(df_store.shape))
    return df, df_store


def load_data_google():
    # [START download-data]

    # gsutil outputs everything to stderr so we need to divert it to stdout.
    subprocess.check_call(['mkdir', '-p', local_data_dir], stderr=sys.stdout)
    subprocess.check_call(
        ['gsutil', 'cp', path.join(cloud_data_dir, train_filename), os.path.join(local_data_dir, train_filename)],
        stderr=sys.stdout)
    subprocess.check_call(
        ['gsutil', 'cp', path.join(cloud_data_dir, test_filename), os.path.join(local_data_dir, test_filename)],
        stderr=sys.stdout)
    subprocess.check_call(
        ['gsutil', 'cp', path.join(cloud_data_dir, store_filename), os.path.join(local_data_dir, store_filename)],
        stderr=sys.stdout)
    subprocess.check_call(
        ['gsutil', 'cp', path.join(cloud_data_dir, feat_matrix_pkl), path.join(local_data_dir, feat_matrix_pkl)],
        stderr=sys.stdout)
    subprocess.check_call(
        ['gsutil', 'cp', path.join(cloud_data_dir, feat_file), path.join(local_data_dir, feat_file)],
        stderr=sys.stdout)
    # [END download-data]

    df_train = pd.read_csv(os.path.join(local_data_dir, train_filename),
                           parse_dates=['Date'],
                           date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                           low_memory=False)

    df_test = pd.read_csv(os.path.join(local_data_dir, test_filename),
                          parse_dates=['Date'],
                          date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                          low_memory=False)

    df_store = pd.read_csv(os.path.join(local_data_dir, store_filename))

    df_train['Type'] = 'train'
    df_test['Type'] = 'test'

    # Combine train and test set
    frames = [df_train, df_test]
    df = pd.concat(frames, sort=True)

    log.info("data loaded:", df.shape)
    log.info("store loaded:", df_store.shape)
    return df, df_store


def extract_recent_data(df_raw, features_x):
    df = df_raw.copy()
    df['Holiday'] = (df['SchoolHoliday'].isin([1])) | (df['StateHoliday'].isin(['a', 'b', 'c']))

    avg_sales, avg_customers = calculate_avg(df)
    sales_last_year, customers_last_year = calculate_avg(
        df.loc[(df['DateInt'] > 20140731) & (df['DateInt'] <= 20150615)])
    sales_last_3months, customers_last_3months = calculate_avg(
        df.loc[(df['DateInt'] > 20150315) & (df['DateInt'] <= 20150615)])
    sales_last_week, customers_last_week = calculate_avg(
        df.loc[(df['DateInt'] > 20150608) & (df['DateInt'] <= 20150615)])
    sales_last_day, customers_last_day = calculate_avg(df.loc[df['DateInt'] == 20150614])
    avg_sales_school_holiday, avg_customers_school_holiday = calculate_avg(df.loc[(df['SchoolHoliday'] == 1)])
    avg_sales_promo, avg_customers_promo = calculate_avg(df.loc[(df['Promo'] == 1)])
    holidays = calculate_holidays(df)

    df = pd.merge(df, avg_sales.reset_index(name='AvgSales'), how='left', on=['Store'])
    df = pd.merge(df, avg_customers.reset_index(name='AvgCustomers'), how='left', on=['Store'])

    df = pd.merge(df, sales_last_year.reset_index(name='AvgYearSales'), how='left', on=['Store'])
    df = pd.merge(df, customers_last_year.reset_index(name='AvgYearCustomers'), how='left', on=['Store'])

    df = pd.merge(df, sales_last_3months.reset_index(name='Avg3MonthsSales'), how='left', on=['Store'])
    df = pd.merge(df, customers_last_3months.reset_index(name='Avg3MonthsCustomers'), how='left', on=['Store'])

    df = pd.merge(df, sales_last_week.reset_index(name='AvgWeekSales'), how='left', on=['Store'])
    df = pd.merge(df, customers_last_week.reset_index(name='AvgWeekCustomers'), how='left', on=['Store'])

    df = pd.merge(df, sales_last_day.reset_index(name='LastDaySales'), how='left', on=['Store'])
    df = pd.merge(df, customers_last_day.reset_index(name='LastDayCustomers'), how='left', on=['Store'])

    df = pd.merge(df, avg_sales_school_holiday.reset_index(name='AvgSchoolHoliday'), how='left', on=['Store'])
    df = pd.merge(df, avg_customers_school_holiday.reset_index(name='AvgCustomersSchoolHoliday'), how='left',
                  on=['Store'])

    df = pd.merge(df, avg_sales_promo.reset_index(name='AvgPromo'), how='left', on=['Store'])
    df = pd.merge(df, avg_customers_promo.reset_index(name='AvgCustomersPromo'), how='left', on=['Store'])

    df = pd.merge(df, holidays, how='left', on=['Store', 'Date'])

    features_x.append("AvgSales")
    features_x.append("AvgCustomers")
    features_x.append("AvgYearSales")
    features_x.append("AvgYearCustomers")
    features_x.append("Avg3MonthsSales")
    features_x.append("Avg3MonthsCustomers")
    features_x.append("AvgWeekSales")
    features_x.append("AvgWeekCustomers")
    features_x.append("LastDaySales")
    features_x.append("LastDayCustomers")
    features_x.append("AvgSchoolHoliday")
    features_x.append("AvgCustomersSchoolHoliday")
    features_x.append("AvgPromo")
    features_x.append("AvgCustomersPromo")
    features_x.append("HolidayLastWeek")
    features_x.append("HolidayNextWeek")

    return df, features_x


def calculate_avg(df):
    store_sales = df.groupby(['Store'])['Sales'].median()
    store_customers = df.groupby(['Store'])['Customers'].median()

    return store_sales, store_customers


def calculate_moving_avg(df, day_window, label):
    store_sales = pd.DataFrame(columns=['Store', 'Avg' + label + 'Sales', 'Date'])
    store_customers = pd.DataFrame(columns=['Store', 'Avg' + label + 'Customers', 'Date'])
    for store in df['Store'].unique():
        df_store = df[df['Store'] == store].set_index('Date')
        sales = df_store.rolling(day_window, min_periods=1)['Sales'].median()
        sales = sales.fillna(sales.median())
        customers = df_store.rolling(day_window, min_periods=1)['Customers'].median()
        customers = customers.fillna(customers.median())
        sales_df = pd.DataFrame()
        sales_df['Avg' + label + 'Sales'] = sales
        sales_df['Store'] = store
        sales_df['Date'] = sales.index
        store_sales = store_sales.append(sales_df, ignore_index=True, sort=False)
        store_sales['Store'] = store_sales.Store.astype(int)
        customers_df = pd.DataFrame()
        customers_df['Avg' + label + 'Customers'] = customers
        customers_df['Store'] = store
        customers_df['Date'] = customers.index
        store_customers = store_customers.append(customers_df, ignore_index=True, sort=False)
        store_customers['Store'] = store_sales.Store.astype(int)
    return store_sales, store_customers


def calculate_holidays(df):
    stores = []
    for store in df['Store'].unique():
        df_store = df[df['Store'] == store].set_index('Date')
        df_store['HolidayLastWeek'] = df_store.rolling(7, min_periods=1)['Holiday'].sum()
        df_store_inverse = df_store.iloc[::-1]
        df_store['HolidayNextWeek'] = df_store_inverse.rolling(7, min_periods=1)['Holiday'].sum()
        stores.append(df_store[['Store', 'HolidayLastWeek', 'HolidayNextWeek']])
    return pd.concat(stores)


def already_extracted():
    file = Path(path.join(local_data_dir, feat_matrix_pkl))
    return file.is_file()


def extract_features(df_raw=None, df_store_raw=None):
    feature_y = 'SalesLog'
    if already_extracted():
        df = pd.read_pickle(path.join(local_data_dir, feat_matrix_pkl))
        return df, read_features(), feature_y

    start_time = time.time()
    df_sales, sales_features, sales_y = extract_sales_feat(df_raw)
    df_sales = df_sales.reset_index()
    log.info('extract_sales_feat: done')
    df_sales.loc[(df_sales['DateInt'] > 20150615) & (df_sales['Type'] == 'train'), 'Type'] = 'validation'
    # outliers
    process_outliers(df_sales)
    log.info('process_outliers: done')
    df_sales, sales_features = extract_recent_data(df_sales, sales_features)
    log.info('extract_recent_data: done')
    df_store, store_features = extract_store_feat(df_store_raw)
    log.info('extract_store_feat: done')
    # construct the feature matrix
    feat_matrix = pd.merge(df_sales[list(set(sales_features + sales_y))], df_store[store_features], how='left',
                           on=['Store'])
    log.info('construct feature matrix: done')

    features_x = selected_features(sales_features, store_features)
    log.info('selected_features: done')

    # dummy code
    feat_matrix['StateHoliday'] = feat_matrix['StateHoliday'].astype('category').cat.codes
    feat_matrix['SchoolHoliday'] = feat_matrix['SchoolHoliday'].astype('category').cat.codes

    check_missing(feat_matrix, features_x, feature_y)
    log.info('check_missing: done')
    check_inf(feat_matrix, features_x, feature_y)
    log.info('check_inf: done')

    log.info("all features: {}".format(features_x))
    log.info("target: {}".format(feature_y))
    log.info("feature matrix dimension: {}".format(feat_matrix.shape))

    feat_matrix.to_pickle(path.join(local_data_dir, feat_matrix_pkl))
    features_to_file(features_x)
    log.info("--- %.2f minutes ---" % ((time.time() - start_time) / 60))

    return feat_matrix, features_x, feature_y


def features_to_file(features_x):
    with open(path.join(local_data_dir, feat_file), 'wb') as f:
        pickle.dump(features_x, f)


def read_features():
    with open(path.join(local_data_dir, feat_file), 'rb') as f:
        return pickle.load(f)


def selected_features(sales_features, store_features):
    features_x = list(set(sales_features + store_features))
    features_x.remove("Type")
    features_x.remove("Store")
    features_x.remove("Id")
    features_x.remove("DateInt")
    return features_x


def check_missing(feat_matrix, features_x, feature_y):
    for feature in [feature_y] + features_x:
        missing = feat_matrix[feature].isna().sum()
        if missing > 0:
            log.info("missing value of {} : {}".format(feature, missing))


def check_inf(feat_matrix, features_x, feature_y):
    for feature in [feature_y] + features_x:
        X = feat_matrix[feature]
        if (X.dtype.char in np.typecodes['AllFloat']
                and not np.isfinite(X.sum())
                and not np.isfinite(X).all()):
            log.info("INF value of {}".format(feature))


def competition_open_datetime(line):
    date = '{}-{}'.format(int(line['CompetitionOpenSinceYear']), int(line['CompetitionOpenSinceMonth']))
    return (pd.to_datetime("2015-08-01") - pd.to_datetime(date)).days


def competition_dist(line):
    return np.log(int(line['CompetitionDistance']))


def promo2_weeks(line):
    if np.isnan(line['Promo2SinceYear']):
        return - 1
    return (2015 - line['Promo2SinceYear']) * 52 + line['Promo2SinceWeek']


def extract_store_feat(df_store_raw):
    df_store = df_store_raw.copy()
    # Convert store type and Assortment to numerical categories
    df_store['StoreType'] = df_store['StoreType'].astype('category').cat.codes
    df_store['Assortment'] = df_store['Assortment'].astype('category').cat.codes
    # Convert competition open year and month to int
    df_store['CompetitionOpenSince'] = df_store.apply(lambda row: competition_open_datetime(row), axis=1).astype(
        np.int64)
    df_store['CompetitionDistance'] = df_store.apply(lambda row: competition_dist(row), axis=1).astype(np.int64)
    # weeks since promo2
    df_store['Promo2Weeks'] = df_store.apply(lambda row: promo2_weeks(row), axis=1).astype(np.int64)
    # TODO how to use PromoInterval
    store_features = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSince', 'Promo2Weeks']
    return df_store, store_features


def extract_sales_feat(df_raw):
    df = df_raw.copy()

    features_x = ['Store', 'Date', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'StateHoliday', 'Type']
    features_y = ['SalesLog', 'Sales', 'Customers']

    # log scale
    df.loc[(df['Type'] == 'train'), 'SalesLog'] = np.log1p(df.loc[(df['Type'] == 'train')]['Sales'])
    # date features
    date_feat = pd.Index(df['Date'])
    df['Week'] = date_feat.week
    df['Month'] = date_feat.month
    df['Year'] = date_feat.year
    df['DayOfMonth'] = date_feat.day
    df['DayOfYear'] = date_feat.dayofyear
    df['Week'] = df['Week'].fillna(0)
    df['Month'] = df['Month'].fillna(0)
    df['Year'] = df['Year'].fillna(0)
    df['DayOfMonth'] = df['DayOfMonth'].fillna(0)
    df['DayOfYear'] = df['DayOfYear'].fillna(0)
    df['DateInt'] = date_feat.year * 10000 + date_feat.month * 100 + date_feat.day
    features_x.remove('Date')
    features_x.append('Week')
    features_x.append('Month')
    features_x.append('Year')
    features_x.append('DayOfMonth')
    features_x.append('DayOfYear')
    features_x.append('DateInt')
    features_x.append('Id')

    return df, features_x, features_y


def mad_based_outlier(points, thresh=3.5):
    if points.empty:
        return False
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    if med_abs_deviation == 0:
        return False

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def to_weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    # convert to sales value as the y are in log scale
    y = np.expm1(y)
    yhat = np.expm1(yhat)
    w = to_weight(y)
    return np.sqrt(np.mean(w * (y - yhat) ** 2))


def rmspe_score(y, yhat):
    w = to_weight(y)
    return np.sqrt(np.mean(w * (y - yhat) ** 2))


def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.expm1(y)
    yhat = np.expm1(yhat)
    w = to_weight(y)
    result = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", result


def process_outliers(df):
    dfs = []
    for store in df['Store'].unique():
        dfs.append(df[df['Store'] == store].copy())
    p = Pool(8)
    df_stores = p.map(process_outliers_store, dfs)
    new_df = pd.concat(df_stores)

    outliers_df = new_df[new_df['Outlier']]
    log.info("size of outliers: {}".format(outliers_df.shape))
    df[new_df.columns] = new_df[new_df.columns]
    df['Outlier'] = new_df['Outlier']


def process_outliers_store(df):
    type_outlier = (df['Type'] == 'train')
    df.loc[df.index, 'Outlier'] = False
    df.loc[type_outlier, 'Outlier'] = mad_based_outlier(df[type_outlier]['Sales'], 3)
    df.loc[(df['Sales'] == 0) & type_outlier, 'Outlier'] = True
    # fill outliers with average sale
    df.loc[(df['Outlier']) & type_outlier, 'SalesLog'] = np.log1p(df[df['Sales'] > 0]['Sales'].median())
    df.loc[(df['Outlier']) & type_outlier, 'Sales'] = df[df['Sales'] > 0]['Sales'].median()
    return df

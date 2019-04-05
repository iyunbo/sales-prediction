import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

seed = 42
feat_matrix_pkl = 'data/feat_matrix.pkl'
feat_file = 'data/features_x.txt'


def load_data(debug=False):
    if already_extracted():
        print("features are previously extracted")
        print("features:", read_features())
        return None, None

    lines = 100 if debug else None
    df_train = pd.read_csv('data/train.csv',
                           parse_dates=['Date'],
                           date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                           nrows=lines,
                           low_memory=False)

    df_test = pd.read_csv('data/test.csv',
                          parse_dates=['Date'],
                          date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                          nrows=lines,
                          low_memory=False)

    df_store = pd.read_csv('data/store_comp.csv', nrows=lines)

    df_train['Type'] = 'train'
    df_test['Type'] = 'test'

    # Combine train and test set
    frames = [df_train, df_test]
    df = pd.concat(frames, sort=True)

    print("data loaded:", df.shape)
    print("store loaded:", df_store.shape)
    return df, df_store


def load_data_google():
    # [START download-data]
    train_filename = 'train.csv'
    test_filename = 'test.csv'
    store_filename = 'store_comp.csv'
    data_dir = 'gs://sales-prediction-iyunbo-mlengine/data'
    local_data_dir = 'data'

    # gsutil outputs everything to stderr so we need to divert it to stdout.
    subprocess.check_call(['mkdir', '-p', local_data_dir], stderr=sys.stdout)
    subprocess.check_call(
        ['gsutil', 'cp', os.path.join(data_dir, train_filename), os.path.join(local_data_dir, train_filename)],
        stderr=sys.stdout)
    subprocess.check_call(
        ['gsutil', 'cp', os.path.join(data_dir, test_filename), os.path.join(local_data_dir, test_filename)],
        stderr=sys.stdout)
    subprocess.check_call(
        ['gsutil', 'cp', os.path.join(data_dir, store_filename), os.path.join(local_data_dir, store_filename)],
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

    print("data loaded:", df.shape)
    print("store loaded:", df_store.shape)
    return df, df_store


def extract_recent_data(df_raw, features_x):
    df = df_raw.copy()
    df['Holiday'] = (df['SchoolHoliday'].isin([1])) | (df['StateHoliday'].isin(['a', 'b', 'c']))

    avg_sales, avg_customers = calculate_avg(df)
    avg_sales_school_holiday, avg_customers_school_holiday = calculate_avg(df.loc[(df['SchoolHoliday'] == 1)])
    avg_sales_state_holiday, avg_customers_state_holiday = calculate_avg(
        df.loc[df['StateHoliday'].isin(['a', 'b', 'c'])])
    avg_sales_promo, avg_customers_promo = calculate_avg(df.loc[(df['Promo'] == 1)])
    holidays = calculate_holidays(df)

    df = pd.merge(df, avg_sales.reset_index(name='AvgSales'), how='left', on=['Store'])
    df = pd.merge(df, avg_customers.reset_index(name='AvgCustomers'), how='left', on=['Store'])
    df = pd.merge(df, avg_sales_school_holiday.reset_index(name='AvgSchoolHoliday'), how='left', on=['Store'])
    df = pd.merge(df, avg_customers_school_holiday.reset_index(name='AvgCustomersSchoolHoliday'), how='left',
                  on=['Store'])
    df = pd.merge(df, avg_sales_state_holiday.reset_index(name='AvgStateHoliday'), how='left', on=['Store'])
    df = pd.merge(df, avg_customers_state_holiday.reset_index(name='AvgCustomersStateHoliday'), how='left',
                  on=['Store'])
    df = pd.merge(df, avg_sales_promo.reset_index(name='AvgPromo'), how='left', on=['Store'])
    df = pd.merge(df, avg_customers_promo.reset_index(name='AvgCustomersPromo'), how='left', on=['Store'])
    df = pd.merge(df, holidays, how='left', on=['Store', 'Date'])

    features_x.append("AvgSales")
    features_x.append("AvgCustomers")
    features_x.append("AvgSchoolHoliday")
    features_x.append("AvgCustomersSchoolHoliday")
    features_x.append("AvgStateHoliday")
    features_x.append("AvgCustomersStateHoliday")
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
    file = Path(feat_matrix_pkl)
    return file.is_file()


def extract_features(df_raw, df_store_raw):
    feature_y = 'SalesLog'
    if already_extracted():
        df = pd.read_pickle(feat_matrix_pkl)
        return df, read_features(), feature_y

    df_sales, sales_features, sales_y = extract_sales_feat(df_raw)
    print('extract_sales_feat: done')
    df_sales.loc[(df_sales['DateInt'] > 20150615) & (df_sales['Type'] == 'train'), 'Type'] = 'validation'
    df_sales, sales_features = extract_recent_data(df_sales, sales_features)
    print('extract_recent_data: done')
    df_store, store_features = extract_store_feat(df_store_raw)
    print('extract_store_feat: done')
    # construct the feature matrix
    feat_matrix = pd.merge(df_sales[list(set(sales_features + sales_y))], df_store[store_features], how='left',
                           on=['Store'])
    print('construct feature matrix: done')

    features_x = selected_features(sales_features, store_features)
    print('selected_features: done')
    check_missing(feat_matrix, features_x)
    print('check_missing: done')
    process_outliers(feat_matrix)
    print('process_outliers: done')

    # dummy code
    feat_matrix['StateHoliday'] = feat_matrix['StateHoliday'].astype('category').cat.codes
    feat_matrix['SchoolHoliday'] = feat_matrix['SchoolHoliday'].astype('category').cat.codes

    print("all features:", features_x)
    print("target:", feature_y)
    print("feature matrix dimension:", feat_matrix.shape)

    feat_matrix.to_pickle(feat_matrix_pkl)
    features_to_file(features_x)

    return feat_matrix, features_x, feature_y


def features_to_file(features_x):
    with open(feat_file, 'wb') as f:
        pickle.dump(features_x, f)


def read_features():
    with open(feat_file, 'rb') as f:
        return pickle.load(f)


def selected_features(sales_features, store_features):
    features_x = list(set(sales_features + store_features))
    features_x.remove("Type")
    features_x.remove("Store")
    features_x.remove("Id")
    return features_x


def check_missing(feat_matrix, features_x):
    for feature in features_x:
        missing = feat_matrix[feature].isna().sum()
        if missing > 0:
            print("missing value of", feature, missing)


def competition_open_datetime(line):
    date = '{}-{}'.format(int(line['CompetitionOpenSinceYear']), int(line['CompetitionOpenSinceMonth']))
    return (pd.to_datetime("2015-08-01") - pd.to_datetime(date)).days


def competition_dist(line):
    return np.log(int(line['CompetitionDistance']))


def promo2_weeks(line):
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
    df_store['Promo2SinceYear'] = df_store['Promo2SinceYear'].fillna(2015)
    df_store['Promo2SinceWeek'] = df_store['Promo2SinceWeek'].fillna(0)
    df_store['Promo2Weeks'] = df_store.apply(lambda row: promo2_weeks(row), axis=1).astype(np.int64)
    # TODO how to use PromoInterval
    store_features = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSince', 'Promo2Weeks']
    return df_store, store_features


def extract_sales_feat(df_raw):
    # Remove rows where store is open, but no sales.
    df = df_raw.loc[~((df_raw['Open'] == 1) & (df_raw['Sales'] == 0))].copy()

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
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = to_weight(y)
    return np.sqrt(np.mean(w * (y - yhat) ** 2))


def rmspe_score(y, yhat):
    w = to_weight(y)
    return np.sqrt(np.mean(w * (y - yhat) ** 2))


def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = to_weight(y)
    result = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", result


def process_outliers(df):
    # consider outliers with mad >= 3
    for store in df['Store'].unique():
        df.loc[(df['Type'] == 'train') & (df['Store'] == store), 'Outlier'] = \
            mad_based_outlier(df.loc[(df['Type'] == 'train') & (df['Store'] == store)]['Sales'], 3)
    # fill outliers with average sale
    outlier_df = df.loc[(df['Type'] == 'train') & (df['Outlier'])]
    df.loc[(df['Type'] == 'train') & (df['Outlier']), 'SalesLog'] = np.log1p(outlier_df['AvgSales'])
    df.loc[(df['Type'] == 'train') & (df['Outlier']), 'Sales'] = outlier_df['AvgSales']

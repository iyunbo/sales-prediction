import logging as log
import os
import os.path as path
import pickle
import subprocess
import sys
import time
from datetime import timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from outliers import smirnov_grubbs as grubbs

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

    log.info('extract_recent_data: done')

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


def extract_refurbishment_events(df):
    df['WasRefurbishments'] = 0
    df['SoonRefurbishments'] = 0
    closed_dates = df[df['Open'] == 0]['Date'].sort_values(ascending=True)
    refurbishment_ends = find_event_ends(closed_dates)
    refurbishment_starts = find_event_starts(closed_dates, refurbishment_ends)
    for end in refurbishment_ends:
        for delta in [1, 2, 3, 4, 5]:
            df.loc[(df['Date'] == end + timedelta(days=delta)) & (df['Open'] == 0), 'WasRefurbishments'] = 1
    for start in refurbishment_starts:
        for delta in [-1, -2, -3, -4, -5]:
            df.loc[(df['Date'] == start + timedelta(days=delta)) & (df['Open'] == 0), 'SoonRefurbishments'] = 1
    return df


def find_event_start(dates, end):
    inverse_dates = np.flip(dates)
    start = None
    target = end
    for date in inverse_dates:
        if date.date() == target.date():
            start = date
            target = target - timedelta(days=1)
        else:
            if not pd.isna(start):
                return start

    if not pd.isna(start):
        return start
    else:
        raise ValueError("Not Found: {} from {}".format(end, dates))


def find_event_starts(dates, end_dates):
    event_starts = []
    for end in end_dates:
        if not pd.isna(end):
            start = find_event_start(dates, end)
            event_starts.append(start)
    return event_starts


def find_event_ends(closed_dates, min_continued_days=5):
    prev_date = pd.to_datetime('2013-01-01')
    count = 0
    event_ends = []
    for date in closed_dates:
        if date == prev_date + timedelta(days=1):
            count = count + 1
        else:
            if count >= min_continued_days:
                event_ends.append(prev_date)
            count = 0
        prev_date = date
    return event_ends


def extract_events_feat(feat_matrix, sales_features):
    log.info("extracting events feature")

    new_df = folk_join(feat_matrix, extract_months_since_promo2)
    feat_matrix['MonthsSincePromo2'] = new_df['MonthsSincePromo2']
    sales_features.append('MonthsSincePromo2')

    new_df = folk_join(feat_matrix, extract_refurbishment_events)
    feat_matrix['SoonRefurbishments'] = new_df['SoonRefurbishments']
    feat_matrix['WasRefurbishments'] = new_df['WasRefurbishments']
    sales_features.append('SoonRefurbishments')
    sales_features.append('WasRefurbishments')

    log.info("extracting events feature: done")
    return feat_matrix, sales_features


months_dict = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sept': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}


def calculate_months_since_promo2(row):
    default_val = 0
    if row['Promo2'] == 0:
        return default_val
    if row['Promo2Weeks'] < 0:
        return default_val
    intervals = map(lambda mon: months_dict[mon], row['PromoInterval'].split(sep=','))
    month_num = row['Month']
    months_since = map(lambda mon: month_num - mon, intervals)
    least_months_since = set(filter(lambda mon: mon >= 0, months_since))
    if len(least_months_since) == 0:
        return default_val
    least_months_since = min(least_months_since)

    return (3 - least_months_since) * row['AvgSales']


def extract_months_since_promo2(df):
    df['MonthsSincePromo2'] = df.apply(lambda row: calculate_months_since_promo2(row), axis=1).astype(np.int64)
    return df


def extract_features(df_raw=None, df_store_raw=None):
    feature_y = 'SalesLog'
    if already_extracted():
        df = pd.read_pickle(path.join(local_data_dir, feat_matrix_pkl))
        return df, read_features(), feature_y

    start_time = time.time()
    df_sales, sales_features, sales_y = extract_sales_feat(df_raw)

    train_validation_limit = 20150615
    df_sales.loc[(df_sales['DateInt'] > train_validation_limit) & (df_sales['Type'] == 'train'), 'Type'] = 'validation'

    process_outlier_sales(df_sales)

    df_sales, sales_features = extract_recent_data(df_sales, sales_features)

    df_store, store_features = extract_store_feat(df_store_raw)

    feat_matrix = construct_feat_matrix(df_sales, df_store, sales_features, sales_y, store_features)

    feat_matrix, sales_features = extract_events_feat(feat_matrix, sales_features)

    features_x = selected_features(sales_features, store_features)

    dummy_encode(feat_matrix)

    check_missing(feat_matrix, features_x)

    check_inf(feat_matrix, features_x, feature_y)

    show_prepared_data(feat_matrix, feature_y, features_x)

    check_outliers(feat_matrix[features_x + [feature_y, 'Store', 'Type']])

    feat_matrix.to_pickle(path.join(local_data_dir, feat_matrix_pkl))
    features_to_file(features_x)

    log.info("--- %.2f minutes ---" % ((time.time() - start_time) / 60))

    return feat_matrix, features_x, feature_y


def construct_feat_matrix(df_sales, df_store, sales_features, sales_y, store_features):
    # construct the feature matrix
    feat_matrix = pd.merge(df_sales[list(set(sales_features + sales_y + ['Outlier']))], df_store[store_features],
                           how='left',
                           on=['Store'])
    log.info('feature matrix constructed')
    return feat_matrix


def show_prepared_data(feat_matrix, feature_y, features_x):
    log.info("all features: {}".format(features_x))
    log.info("target: {}".format(feature_y))
    log.info("feature matrix dimension: {}".format(feat_matrix.shape))


def dummy_encode(feat_matrix):
    # dummy code
    feat_matrix['StateHoliday'] = feat_matrix['StateHoliday'].astype('category').cat.codes
    feat_matrix['SchoolHoliday'] = feat_matrix['SchoolHoliday'].astype('category').cat.codes


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
    features_x.remove('Date')
    features_x.remove('Open')
    features_x.remove('Promo2')
    features_x.remove('PromoInterval')
    log.info('selected_features: done')
    return features_x


def check_missing(feat_matrix, features_x):
    for feature in features_x:
        missing = feat_matrix[feature].isna().sum()
        if missing > 0:
            raise ValueError("missing value of {} : {}".format(feature, missing))
    log.info('check_missing: done')


def check_inf(feat_matrix, features_x, feature_y):
    for feature in [feature_y] + features_x:
        vec = feat_matrix[feature]
        if (vec.dtype.char in np.typecodes['AllFloat']
                and not np.isfinite(vec.sum())
                and not np.isfinite(vec).all()):
            raise ValueError("INF value of {}".format(feature))
    log.info('check_inf: done')


def competition_open_datetime(line):
    date = '{}-{}'.format(int(line['CompetitionOpenSinceYear']), int(line['CompetitionOpenSinceMonth']))
    return (pd.to_datetime("2015-08-01") - pd.to_datetime(date)).days


def competition_dist(line):
    return np.log1p(int(line['CompetitionDistance']))


def promo2_weeks(line):
    if np.isnan(line['Promo2SinceYear']):
        return - 1
    return (2015 - line['Promo2SinceYear']) * 52 + line['Promo2SinceWeek']


def is_of_value(row, feature, value):
    if row[feature] == value:
        return 1
    return 0


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
    store_features = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSince', 'Promo2Weeks',
                      'Promo2', 'PromoInterval']

    log.info('extract_store_feat: done')
    return df_store, store_features


def is_near_christmas(row, days):
    if row['Month'] != 12:
        return 0
    if days < 0:
        diff_days = 25 - row['DayOfMonth']
    else:
        diff_days = row['DayOfMonth'] - 25
    return 1 if abs(days) >= diff_days >= 0 else 0


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
    df['IsSaturday'] = df.apply(lambda row: is_of_value(row, 'DayOfWeek', 6), axis=1).astype(np.int64)
    df['IsSunday'] = df.apply(lambda row: is_of_value(row, 'DayOfWeek', 7), axis=1).astype(np.int64)
    df['SoonChristmas'] = df.apply(lambda row: is_near_christmas(row, -5), axis=1).astype(np.int64)
    df['WasChristmas'] = df.apply(lambda row: is_near_christmas(row, 5), axis=1).astype(np.int64)
    # features_x.remove('Date')
    features_x.append('Week')
    features_x.append('Month')
    features_x.append('Year')
    features_x.append('DayOfMonth')
    features_x.append('DayOfYear')
    features_x.append('DateInt')
    features_x.append('IsSaturday')
    features_x.append('IsSunday')
    features_x.append('SoonChristmas')
    features_x.append('WasChristmas')
    features_x.append('Open')
    features_x.append('Id')
    log.info('extract_sales_feat: done')

    return df.reset_index(), features_x, features_y


def to_weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    # convert to sales value as the y are in log scale
    y = np.expm1(y)
    # https://www.kaggle.com/c/rossmann-store-sales/discussion/17601
    yhat = np.expm1(yhat) * 0.985
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


def process_outlier_sales(df):
    new_df = folk_join(df, fill_outlier_sales)

    outliers_df = new_df[new_df['Outlier']]
    log.info("size of outliers: {}".format(outliers_df.shape))
    df[new_df.columns] = new_df[new_df.columns]

    log.info('process_outliers: done')


def folk_join(df, func, n_jobs=8, by_feat='Store'):
    folks = []
    for feature_val in df[by_feat].unique():
        folks.append(df[df[by_feat] == feature_val].copy())
    process_pool = Pool(n_jobs)
    df_stores = process_pool.map(func, folks)
    joined = pd.concat(df_stores, sort=False)
    return joined


def fill_outlier_sales(df):
    type_outlier = (df['Type'] == 'train')
    # by default, values are not outlier
    df.loc[df.index, 'Outlier'] = False
    # detect outliers by test of grubbs
    outlier_idx = grubbs.two_sided_test_indices(df[type_outlier]['SalesLog'], 0.05)
    # tag outliers
    df.loc[df.index[outlier_idx], 'Outlier'] = True
    df.loc[(df['SalesLog'] == 0) & type_outlier, 'Outlier'] = True
    # fill outliers with average sale
    df.loc[(df['Outlier']) & type_outlier, 'SalesLog'] = df[(df['SalesLog'] > 0)]['SalesLog'].median()
    df.loc[(df['Outlier']) & type_outlier, 'Sales'] = df[(df['Sales'] > 0)]['Sales'].median()
    return df


def check_outliers(df):
    log.info("checking outliers with test of grubbs")
    new_df = folk_join(df[df['Type'] == 'train'], test_grubbs)
    new_df.to_csv(path.join(local_data_dir, 'outliers.csv'))
    log.info("checking outliers: done, total outliers: {}".format(new_df['Count'].sum()))


def test_grubbs(df):
    outliers = pd.DataFrame(columns=['Store', 'Column', 'Count'])
    i = 0
    store = df['Store'].iloc[0]
    columns = list(df.columns)
    for col in ['StateHoliday', 'Type', 'SalesLog', 'IsSunday', 'IsSaturday']:
        columns.remove(col)

    for num_col in columns:
        outlier_idx = grubbs.two_sided_test_indices(df[num_col], 0.05)
        outlier_count = len(outlier_idx)
        if outlier_count > 0:
            outliers.loc[i] = [store, num_col, outlier_count]
            i = i + 1

    return outliers

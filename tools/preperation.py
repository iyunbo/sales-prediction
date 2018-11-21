import pandas as pd
import numpy as np
from pathlib import Path
import pickle

seed = 42
feat_matrix_pkl = '../data/feat_matrix.pkl'
feat_file = '../data/features_x.txt'


def load_data(debug=False):
    if already_extracted():
        print("features are previously extracted")
        return None, None

    lines = 100 if debug else None
    df_train = pd.read_csv('../data/train.csv',
                           parse_dates=['Date'],
                           date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                           nrows=lines,
                           low_memory=False)

    df_test = pd.read_csv('../data/test.csv',
                          parse_dates=['Date'],
                          date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                          nrows=lines,
                          low_memory=False)

    df_store = pd.read_csv('../data/store.csv', nrows=lines)

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
    sales_per_day, customers_per_day = recent_features(
        df.loc[(df['DateInt'] <= 20150615)])
    sales_per_day_last_3m, customers_per_day_last_3m = recent_features(
        df.loc[(df['DateInt'] > 20150315) & (df['DateInt'] <= 20150615)])
    sales_per_day_last_month, customers_per_day_last_month = recent_features(
        df.loc[(df['DateInt'] > 20150515) & (df['DateInt'] <= 20150615)])
    sales_per_day_last_week, customers_per_day_last_week = recent_features(
        df.loc[(df['DateInt'] > 20150608) & (df['DateInt'] <= 20150615)])

    df = pd.merge(df, sales_per_day.reset_index(name='AvgSales'), how='left', on=['Store'])
    df = pd.merge(df, customers_per_day.reset_index(name='AvgCustomers'), how='left', on=['Store'])
    df = pd.merge(df, sales_per_day_last_3m.reset_index(name='AvgSales3Months'), how='left', on=['Store'])
    df = pd.merge(df, customers_per_day_last_3m.reset_index(name='AvgCustomers3Months'), how='left', on=['Store'])
    df = pd.merge(df, sales_per_day_last_month.reset_index(name='AvgSalesMonth'), how='left', on=['Store'])
    df = pd.merge(df, customers_per_day_last_month.reset_index(name='AvgCustomersMonth'), how='left', on=['Store'])
    df = pd.merge(df, sales_per_day_last_week.reset_index(name='AvgSalesWeek'), how='left', on=['Store'])
    df = pd.merge(df, customers_per_day_last_week.reset_index(name='AvgCustomersWeek'), how='left', on=['Store'])

    features_x.append("AvgSales")
    features_x.append("AvgCustomers")
    features_x.append("AvgSales3Months")
    features_x.append("AvgCustomers3Months")
    features_x.append("AvgSalesMonth")
    features_x.append("AvgCustomersMonth")
    features_x.append("AvgSalesWeek")
    features_x.append("AvgCustomersWeek")

    return df, features_x


def recent_features(df):
    store_sales = df.groupby(['Store'])['Sales'].sum()
    store_customers = df.groupby(['Store'])['Customers'].sum()
    store_days = df.groupby(['Store'])['DateInt'].count()

    return store_sales / store_days, store_customers / store_days


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
    process_missing(feat_matrix, features_x)
    print('process_missing: done')
    process_outliers(feat_matrix)
    print('process_outliers: done')

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
    return features_x


def process_missing(feat_matrix, features_x):
    for feature in features_x:
        feat_matrix[feature] = feat_matrix[feature].fillna(-1)


def competition_open_datetime(line):
    try:
        date = '{}-{}'.format(int(line['CompetitionOpenSinceYear']), int(line['CompetitionOpenSinceMonth']))
        return pd.to_datetime(date)
    except:
        return np.nan


def extract_store_feat(df_store_raw):
    df_store = df_store_raw.copy()
    # Convert store type and Assortment to numerical categories
    df_store['StoreType'] = df_store['StoreType'].astype('category').cat.codes
    df_store['Assortment'] = df_store['Assortment'].astype('category').cat.codes
    # Convert competition open year and month to int
    df_store['CompetitionOpenSince'] = df_store.apply(lambda row: competition_open_datetime(row), axis=1).astype(
        np.int64)
    # exclude Promo2 related features due to irrelevance and high missing ratio
    store_features = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSince']

    return df_store, store_features


def extract_sales_feat(df_raw):
    # Remove rows where store is open, but no sales.
    df = df_raw.loc[~((df_raw['Open'] == 1) & (df_raw['Sales'] == 0))].copy()

    features_x = ['Store', 'Date', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'StateHoliday', 'Type']
    features_y = ['SalesLog', 'Sales']

    # log scale
    df.loc[(df['Type'] == 'train'), 'SalesLog'] = np.log1p(df.loc[(df['Type'] == 'train')]['Sales'])
    # dummy code
    df['StateHoliday'] = df['StateHoliday'].astype('category').cat.codes
    df['SchoolHoliday'] = df['SchoolHoliday'].astype('category').cat.codes
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

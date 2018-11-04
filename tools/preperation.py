import pandas as pd
import numpy as np


def load_data():
    df_train = pd.read_csv('../data/train.csv',
                           parse_dates=['Date'],
                           date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                           low_memory=False)

    df_test = pd.read_csv('../data/test.csv',
                          parse_dates=['Date'],
                          date_parser=(lambda dt: pd.to_datetime(dt, format='%Y-%m-%d')),
                          low_memory=False)

    df_store = pd.read_csv('../data/store.csv')

    df_train['Train'] = True
    df_test['Train'] = False

    # Combine train and test set
    frames = [df_train, df_test]
    df = pd.concat(frames, sort=True)

    print("data loaded:", df.shape)
    print("store loaded:", df_store.shape)
    return df, df_store


def competition_open_datetime(line):
    try:
        date = '{}-{}'.format(int(line['CompetitionOpenSinceYear']), int(line['CompetitionOpenSinceMonth']))
        return pd.to_datetime(date)
    except:
        return np.nan


def extract_features(df_raw, df_store_raw):
    df, sales_features, features_y = extract_sales_feat(df_raw)

    df_store, store_features = extract_store_feat(df_store_raw)

    # construct the feature matrix
    feat_matrix = pd.merge(df[sales_features], df_store[store_features], how='left', on=['Store'])

    # all missing values to -1
    feat_matrix.fillna(-1)

    print("all features:\n", list(set(sales_features + store_features)))
    print("target:", features_y)
    print("feature matrix shape:", feat_matrix.shape)

    return feat_matrix


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

    features_x = ['Store', 'Date', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'StateHoliday', 'Train']
    features_y = ['SalesLog']

    # log scale
    df.loc[df['Train'], 'SalesLog'] = np.log1p(df.loc[df['Train']]['Sales'])
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
    df['DateInt'] = df['Date'].astype(np.int64)
    features_x.remove('Date')
    features_x.append('Week')
    features_x.append('Month')
    features_x.append('Year')
    features_x.append('DayOfMonth')
    features_x.append('DayOfYear')
    features_x.append('DateInt')

    return df, features_x, features_y

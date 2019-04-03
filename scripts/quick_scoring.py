from preparation import load_data, extract_features
from models import quick_score
import datetime, time


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    features_x.remove('AvgCustomersPromo')
    features_x.remove('HolidayLastWeek')
    features_x.remove('HolidayNextWeek')
    score, duration = quick_score(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y)
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(ts)
    print('RMSPE = %.4f' % score)
    print('Duration : %.2f seconds' % duration)


if __name__ == '__main__':
    main()

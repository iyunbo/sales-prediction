from trainer.model import train_xgboost
from trainer.preparation import load_data, extract_features


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    # features_x.remove('WasChristmas')
    train_xgboost(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y)


if __name__ == '__main__':
    main()

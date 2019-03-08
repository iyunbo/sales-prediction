from preperation import load_data, extract_features
from models import tune_random_forest


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    tune_random_forest(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y)


if __name__ == '__main__':
    main()

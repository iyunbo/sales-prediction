from trainer.preparation import load_data, extract_features
from trainer.model import run_models


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    run_models(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y)


if __name__ == '__main__':
    main()

from preperation import load_data, extract_features
from models import train_random_forest, train_xgboost, tune_xgboost


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    params = {'bootstrap': True, 'criterion': 'mse', 'max_depth': 20, 'max_features': 'auto', 'max_leaf_nodes': None,
              'min_impurity_decrease': 0, 'min_impurity_split': None, 'min_samples_leaf': 10, 'min_samples_split': 2,
              'min_weight_fraction_leaf': 0, 'n_estimators': 10, 'n_jobs': 1, 'oob_score': True, 'random_state': None,
              'verbose': 0, 'warm_start': True}
    # train_random_forest(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y, params)
    tune_xgboost(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y)


if __name__ == '__main__':
    main()

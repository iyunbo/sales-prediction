from preperation import load_data, extract_features
from models import train_random_forest


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    # best param from last random search
    # params = {'bootstrap': True, 'criterion': 'mse', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None,
    #           'min_impurity_decrease': 0, 'min_impurity_split': None, 'min_samples_leaf': 10, 'min_samples_split': 2,
    #           'min_weight_fraction_leaf': 0, 'n_estimators': 100, 'n_jobs': 6, 'oob_score': True, 'random_state': None,
    #           'verbose': 0, 'warm_start': True}
    train_random_forest(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y)


if __name__ == '__main__':
    main()

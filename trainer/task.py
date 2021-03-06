from .model import train_ensemble
from .preparation import load_data, extract_features

df, df_store = load_data(debug=False)
feat_matrix, features_x, feature_y = extract_features(df, df_store)
train_ensemble(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y)

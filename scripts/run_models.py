from preperation import load_data, extract_features
from models import run_models

df, df_store = load_data(debug=False)
feat_matrix, features_x, feature_y = extract_features(df, df_store)
run_models(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y)
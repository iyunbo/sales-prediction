from preperation import load_data, extract_features

df, df_store = load_data(debug=True)
feat_matrix = extract_features(df, df_store)


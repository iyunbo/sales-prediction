from preparation import load_data, extract_features


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    # TODO
    # train with the best model and with complete dataset (train + validation)


if __name__ == '__main__':
    main()

import os.path as path

import numpy as np
import xgboost as xgb

from trainer.model import final_model
from trainer.preparation import load_data, extract_features, local_data_dir


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    df_test = feat_matrix.loc[feat_matrix['Type'] == 'test'].copy()
    predict = forecast(df_test, features_x)
    df_test['Id'] = df_test.Id.astype(int)
    df_test = df_test.set_index('Id')
    df_test['Sales'] = predict
    result = df_test[['Sales']].copy()
    print(result.head())
    result.to_csv(path.join(local_data_dir, 'submission.csv'))


def forecast(df_test, features_x):
    model = final_model()
    predict = model.predict(xgb.DMatrix(df_test[features_x]), ntree_limit=416)
    return np.expm1(predict).astype(int)


if __name__ == '__main__':
    main()

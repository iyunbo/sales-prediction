# [START setup]
import datetime
import time

from .model import quick_score
from .preparation import load_data_google, extract_features

# Fill in your Cloud Storage bucket name
BUCKET_NAME = 'sales-prediction-iyunbo-mlengine'
# [END setup]


df, df_store = load_data_google()
feat_matrix, features_x, feature_y = extract_features(df, df_store)
score, duration = quick_score(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x, feature_y)
ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(ts)
print('RMSPE = %.4f' % score)
print('Duration : %.2f seconds' % duration)

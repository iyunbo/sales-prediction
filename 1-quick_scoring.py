import datetime
import time

import numpy as np

from trainer.model import quick_score
from trainer.preparation import load_data, extract_features


def main():
    df, df_store = load_data(debug=False)
    feat_matrix, features_x, feature_y = extract_features(df, df_store)
    score1, score2, score3, duration = quick_score(feat_matrix.loc[~(feat_matrix['Type'] == 'test')], features_x,
                                                   feature_y)
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(ts)
    print('RMSPE = %.4f' % np.mean([score1, score2, score3]))
    print('Duration : %.2f seconds' % duration)


if __name__ == '__main__':
    main()

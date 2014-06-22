import cPickle as pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df_label = []
for i in xrange(8):
  filename = 'label%d.csv' %(i)
  df_label.append(pd.read_csv(filename))

auc = np.vstack([dl['target'].values for dl in df_label]).T
auc_norm = MinMaxScaler().fit_transform(auc)
auc_norm_sum_01234 = auc_norm[:,:5].sum(axis=1)
submission = pickle.load(open('../submission_projectid.pickle', 'rb'))
submission['is_exciting'] = auc_norm_sum_01234
submission.to_csv('submission.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

df_label = []
for i in xrange(8):
  filename = 'label%d.csv' %(i)
  df_label.append(pd.read_csv(filename))

auc = np.vstack([dl['target'].values for dl in df_label]).T
#auc_sum_01234 = auc[:,:5].sum(axis=1)
auc_norm = MinMaxScaler().fit_transform(auc)
auc_norm_sum_01234 = auc_norm[:,:5].sum(axis=1)
submission = pd.DataFrame(df_label[0]['projectid'])
#submission['is_exciting'] = auc_sum_01234
submission['is_exciting'] = auc_norm_sum_01234
submission_filename = os.path.basename(os.getcwd())+'.csv'
submission.to_csv(submission_filename, index=False)

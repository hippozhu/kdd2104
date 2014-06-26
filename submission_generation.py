import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os, sys

df_label = []
for i in xrange(8):
  filename = 'label%d.csv' %(i)
  df_label.append(pd.read_csv(filename))

w0 = float(sys.argv[1]) 
w1 = float(sys.argv[2]) 
w2 = float(sys.argv[3]) 

weights = \
[w0] +\
[w1]*4 + \
[w2]*3
auc = np.vstack([dl['is_exciting'].values for dl in df_label]).T
#auc_sum_01234 = auc[:,:5].sum(axis=1)
auc_norm = MinMaxScaler().fit_transform(auc)
auc_norm_weighted = auc_norm*weights
scores = auc_norm_weighted.sum(axis=1)
submission = pd.DataFrame(df_label[0]['projectid'])
submission['is_exciting'] = scores
submission_filename = '%s_%.2f_%.2f_%.2f.csv' %(os.path.basename(os.getcwd()), w0, w1, w2)
submission.to_csv(submission_filename, index=False)

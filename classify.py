import numpy as np
from time import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression
from sklearn.metrics import roc_auc_score as AUC

def minibatch_generator(data_size, mb_size=100, n_iter=3):
  indices = range(data_size)
  for _ in xrange(n_iter):
    np.random.shuffle(indices)
    for i in xrange(0, data_size, mb_size):
      yield indices[i:min(i+mb_size, data_size)]

def sgd(X_train, y_train, X_validate, y_validate, X_test, cw, alpha, regression=False):
  #cw = 2.5
  if regression:
    clf = SGDRegressor(alpha=alpha)
  else:
    #clf = SGDClassifier(class_weight = {1:cw}, alpha=alpha)
    clf = SGDClassifier(class_weight = {1:cw}, alpha=alpha, loss='log')
  print clf
  training_data_size = y_train.shape[0]
  n_iter = 3
  mb_size = 100
  iter_mb = minibatch_generator(training_data_size, mb_size = mb_size, n_iter = n_iter)
  total = 0
  n_total_batch = n_iter*training_data_size/mb_size
  t0 = time()
  recent_auc = []
  for n_batch, batch in enumerate(iter_mb):
    x, y = X_train[batch], y_train[batch]
    if regression:
      sw = np.ones(y.shape[0])
      sw[np.where(y==1)[0]] = cw
      clf.partial_fit(x, y, sample_weight=sw)
    else:
      clf.partial_fit(x, y, classes = [1, 0])
    total += y.shape[0]
    if (n_batch+1)%1000 == 0:
      if regression:
        #y_pred_validate_val = clf.decision_function(X_validate)
        y_pred_validate_val = clf.predict(X_validate)
      else:
        #y_pred_validate_val = clf.decision_function(X_validate)
        y_pred_validate_val = clf.predict_proba(X_validate)[:,1]
      print 'auc:%.3f, %d samples in %ds (cw: %.2f)' %(AUC(y_validate, y_pred_validate_val), total, time()-t0, cw)
    if n_batch>n_total_batch-100:
      if regression:
        y_pred_validate_val = clf.predict(X_validate)
      else:
        y_pred_validate_val = clf.predict_proba(X_validate)[:,1]
      recent_auc.append(AUC(y_validate, y_pred_validate_val))
  latest_auc_avg = np.mean(recent_auc)
  print 'cw=%.2f, avg auc of last %d bathes: %.3f' %(cw, len(recent_auc), latest_auc_avg)
  if regression:
    return clf.predict(X_test)
  else:
    return clf.predict_proba(X_test)[:,1]

def logit(X_train, y_train, X_validate, y_validate, X_test, cw):
  clf = LogisticRegression(class_weight = {1:cw})
  clf.fit(X_train, y_train)
  y_pred_validate_val = clf.predict(X_validate)
  print 'auc:%.3f' %(AUC(y_validate, y_pred_validate_val))
  
'''
def sgd_regression(X_train, y_train, X_validate, y_validate, X_test, cw):
  #cw = 2.5
  clf = SGDRegressor()
  training_data_size = y_train.shape[0]
  iter_mb = minibatch_generator(training_data_size, n_iter = 3)
  
  total = 0
  t0 = time()
  recent_auc = []
  for n_batch, batch in enumerate(iter_mb):
    x, y = X_train[batch], y_train[batch]
    sw = np.ones(y.shape[0])
    sw[np.where(y==1)[0]] = cw
    clf.partial_fit(x, y, sample_weight=sw)
    #clf.partial_fit(x, y)
    total += y.shape[0]
    if (n_batch+1)%500 == 0:
      y_pred_validate_val = clf.decision_function(X_validate)
      auc = AUC(y_validate, y_pred_validate_val)
      #recent_auc.append(auc)
      #last_avg = np.mean(recent_auc)
      last_avg = auc
      print 'auc:%.3f, recent_aucs:%.3f, %d samples in %ds (cw: %.2f)' %(auc, last_avg, total, time()-t0, cw)
      #recent_auc = []
  print 'cw=%.2f' %cw
  return clf.decision_function(X_test)

def lsvm(X_train, y_train, X_validate, y_validate, X_test):
  clf = SVC(kernel='linear', class_weight = {1:2})
  clf.fit(X_train, y_train)
  y_pred_validate_val = clf.decision_function(X_validate)
  auc = AUC(y_validate, y_pred_validate_val)
  print 'auc by lsvm:', auc
'''

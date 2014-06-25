import numpy as np
from time import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import roc_auc_score as AUC


def minibatch_generator(data_size, mb_size=100, n_iter=3):
  indices = range(data_size)
  for _ in xrange(n_iter):
    np.random.shuffle(indices)
    for i in xrange(0, data_size, mb_size):
      yield indices[i:min(i+mb_size, data_size)]

def sgd_classification(X_train, y_train, X_validate, y_validate, X_test, cw):
  #cw = 2.5
  clf = SGDClassifier(class_weight = {1:cw})
  training_data_size = y_train.shape[0]
  iter_mb = minibatch_generator(training_data_size, n_iter = 3)
  total = 0
  t0 = time()
  recent_auc = []
  for n_batch, batch in enumerate(iter_mb):
    x, y = X_train[batch], y_train[batch]
    clf.partial_fit(x, y, classes = [1, 0])
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
  
if __name__ == '__main__':
  N_TRAINING = 524928
  data = np.loadtxt('data/project_basics_int', delimiter=',', skiprows=1)
  pid_train, labels = load_outcome('data/outcomes_binary_nomissing.csv')
  pid_test = [line.strip() for line in open('test_projects')]
  projects_train, projects_test = load_projects('data/projects_fixed4.csv')
  X = MinMaxScaler().fit_transform(data)
  y = labels[1]
  skf = StratifiedKFold(y, 20)
  folds = [test for train, test in skf]
  train = folds[0]
  test = folds[1]
  weights = np.array([1.0]*train.shape[0])
  pos = np.where(y[train]==1)[0]
  clf = SVC(kernel='linear')
  for w in [2,3,4]:
    print 'weight', w
    weights[pos] = w
    clf.fit(X[train], y[train], sample_weight=weights)
    y_pred = clf.predict(X[test])
    print confusion_matrix(y_pred, y[test])
    print accuracy_score(y_pred, y[test])


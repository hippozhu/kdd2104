# coding: utf-8
import cPickle as pickle
from time import time
from datetime import date
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import ExtraTreesClassifier

from kdd_utility import *
from entities import *

import pandas as pd

def bin_index(n):
  if n>0 and n<=10:
    return n-1
  elif n<=100:
    return (n-1)/10 + 9
  elif n<=150:
    return 19
  elif n<=200:
    return 20
  elif n<=300:
    return 21
  else:
    return 22

def discretize_num(number):
  return reduce(lambda i, j: i + j, map(lambda x: min(int(number/pow(10,x)), 9), xrange(2,6)))

class Projects:
  def __init__(self, outcome_file):
    self.state_feature_index = 7
    self.zip_feature_index = 8
    self.binary_feature_index = [12, 13, 14, 15, 16, 17, 19, 20, 32, 33]
    self.categorical_feature_index = [18, 21, 22, 25, 26, 27, 28]
    self.numerical_feature_index = [29, 30, 31]
    self.date_feature_index = 34
    self.vec = DictVectorizer(sparse=False)
    self.load_projects(outcome_file)
    
  def load_projects(self, outcome_file):
    fin = open(outcome_file)
    self.project_feature_names = fin.next().strip().split(',')
    self.projects = dict((line.strip().split(',')[0], line.strip().split(','))\
    for line in fin)
    fin.close()
    
  def all_features(self, pids):
    measurements_state = map(lambda k: {str(self.state_feature_index): self.projects[k][self.state_feature_index]}, pids)
    measurements_zip = map(lambda k: {str(self.zip_feature_index): self.projects[k][self.zip_feature_index][:3]}, pids)
    measurements_bin = map(lambda k: dict((str(fi), self.projects[k][fi]) for fi in self.binary_feature_index), pids)
    measurements_cat = map(lambda k: dict((str(fi), self.projects[k][fi]) for fi in self.categorical_feature_index), pids)
    #measurements_num = map(lambda k: [float(self.projects[k][fi]) for fi in self.numerical_feature_index], pids)
    measurements_num = map(lambda k: dict((str(fi), str(discretize_num(float(self.projects[k][fi])))) for fi in self.numerical_feature_index), pids)
    return self.vec.fit_transform(measurements_state), self.vec.fit_transform(measurements_zip), self.vec.fit_transform(measurements_bin), self.vec.fit_transform(measurements_cat), self.vec.fit_transform(measurements_num)#,np.array(measurements_num)
 
def feature_importances(measurements, topic, labels, label_names):
  vec = DictVectorizer(sparse=False)
  X = vec.fit_transform(measurements)
  feature_names = vec.get_feature_names()
  forest = ExtraTreesClassifier(n_estimators=250, max_features=None, n_jobs=15, random_state=0)
  for i, y in enumerate(labels):
    label_name = label_names[i]
    print 'fitting', label_name
    t0 = time()
    forest.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    n_features = importances.shape[0]
    #pickle.dump(importances, open('importances', 'wb'))
    pl.figure(figsize=(20,15))
    pl.title("Feature importances w.r.t. %s" %(label_name))
    pl.bar(range(n_features), importances[indices], color="r",align="center")
    #f_name = map(lambda i: feature_names[i].split('=')[1], indices)
    f_name = map(lambda i: feature_names[i], indices)
    pl.xticks(range(n_features), f_name)
    pl.xlim([-1, n_features])
    pl.savefig('.'.join([topic, label_name, 'png']))

def featureAnalysis(feature, feature_pos, label):
  hist, _ = np.histogram(feature, bins=20, range=(0,1))
  print ','.join(str(x) for x in hist)
  print ''
  hist, _ = np.histogram(feature_pos, bins=20, range=(0,1))
  print ','.join(str(x) for x in hist)
  print ''

def dummy_variables_binary(data, cols):
  dummies = [pd.get_dummies(data[variable], prefix=variable)[variable+'_t'] for variable in cols]
  for variable in cols:
    data.drop(variable, axis=1, inplace=True)
  return pd.concat([data]+dummies, axis=1)

def dummy_variables_categorical(data, cols):
  dummies = [pd.get_dummies(data[variable], prefix=variable) for variable in cols]
  for variable in cols:
    data.drop(variable, axis=1, inplace=True)
  return pd.concat([data]+dummies, axis=1)

def normalize_num(data, col, max_value, min_value):
  data[col][data[col]>max_value] = max_value 
  data[col][data[col]<min_value] = min_value 
  data[col+'_scaled'] = data[col]/(max_value-min_value)
  data.drop(col, axis=1, inplace=True)
  return data

if __name__ == '__main__':
  pid_train, labels = load_outcome('data/outcomes_binary_nomissing.csv')
  pid_test = [line.strip() for line in open('test_projects')]
  projects_train, projects_test = load_projects('data/projects_fixed4.csv')

  schools = load_schools('data/schools.csv')
  teachers =  load_teachers('data/teachers.csv')
  outcomes = load_entity('data/outcomes_binary_nomissing.csv', Outcome, True)

  projects_teacher = projects_by_teacher(projects_train, teachers)
  projects_school = projects_by_school(projects_train, schools)


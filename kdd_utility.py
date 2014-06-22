import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab as pl

label_names = ['is_exciting','at_least_1_teacher_referred_donor',\
'fully_funded','at_least_1_green_donation','great_chat',\
'three_or_more_non_teacher_referred_donors',\
'one_non_teacher_referred_donor_giving_100_plus',\
'donation_from_thoughtful_donor']

def load_outcome(outcome_file):
  fin = open(outcome_file)
  fin.next()
  pid = []
  labels = []
  for line in fin:
    contents = line.strip().split(',')
    pid.append(contents[0])
    labels.append(map(lambda x: 1 if x=='t' else 0, contents[1:]))
  return np.asarray(pid), np.array(labels, dtype=np.int32).T

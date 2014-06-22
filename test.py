import cPickle as pickle
from time import time

t = time()
ti = pickle.load(open('train-validate-test', 'rb'))
print time()-t

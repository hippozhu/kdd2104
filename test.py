import numpy as np
import cPickle as pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

def essay_tokens_generator(filename):
  fin = open(filename)
  fin.next()
  for i, line in enumerate(fin):
    if (i+1)%10000 == 0:
      print 'read', i+1
    yield line.strip().split(',')[1]
  fin.close()

#vectorizer = TfidfVectorizer(min_df=5, stop_words='english', max_df=0.5, max_features=1000)
corpus = essay_tokens_generator(sys.argv[1])
#vectors = vectorizer.fit_transform(corpus)
#np.savetxt('tfidf1000.csv', vectors.toarray(), delimiter=',')
t0 = time()
vectorizer = pickle.load(open('tfidf_vectorizer', 'rb'))
vectors = vectorizer.transform(corpus)
print time()-t0
'''
pickle.dump(vectors, open('tfidf1000.pickle', 'wb'))
sX_test = sparse.csr_matrix(X_test)
from scipy.sparse import hstack
hstack((sX_test, test_vectors))
'''

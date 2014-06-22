import csv
import multiprocessing
from essay_feature import *
from kdd_utility import *
from entities import *

NUM_PROCS = multiprocessing.cpu_count()
#train_projects = [line.strip() for line in open('train_projects')]
#train_projects, labels = load_outcome('data/outcomes_binary_nomissing.csv')
#test_projects = [line.strip() for line in open('test_projects')]

class CSVWorker(object):
  def __init__(self, numprocs, infile, outfile):
    self.numprocs = numprocs
    #self.outfile = outfile
    #self.in_csvfile = csv.reader(self.infile)
    self.inq = multiprocessing.Queue()
    self.outq = multiprocessing.Queue()

    self.pin = multiprocessing.Process(target=self.parse_input_csv,\
    args=(infile,))
    self.pout = multiprocessing.Process(target=self.write_output_csv,\
    args=(outfile,))
    self.ps = [multiprocessing.Process(target=self.calc_features, args=())\
    for i in range(self.numprocs)]

    self.pin.start()
    #self.parse_input_csv(infile)
    #self.write_output_csv(outfile)
    self.pout.start()
    for p in self.ps:
	p.start()

    self.pin.join()
    i = 0
    for p in self.ps:
      p.join()
      print "Done", i
      i += 1

    self.pout.join()

  def parse_input_csv(self, inputfile):
    print 'start reading file', multiprocessing.current_process().name
    essayreader = csv.reader(open(inputfile), quotechar='"')
    essayreader.next()
    for i, row in enumerate(essayreader):
      self.inq.put(row)
      if (i+1)%100000 == 0:
	print multiprocessing.current_process().name, i+1
    for i in xrange(self.numprocs):
      self.inq.put('STOP')

  def calc_features(self):
    p_name = multiprocessing.current_process().name
    for i, row in enumerate(iter(self.inq.get, 'STOP')):
      try:
	essay = Essay(row)
	essay.preprocessing()
        #self.outq.put( (row[0], Essay(row).readability_features()) )
        self.outq.put( (row[0], len(essay.essay_tokens)) )
      except ValueError:
        #self.outq.put( (row[0], tuple([-1]*31)) )
        self.outq.put( (row[0], 0) )
        print 'Empty essay at %s' %(row[0])
      if (i+1)%10000 == 0:
	print p_name, i+1
    self.outq.put('STOP')

  def write_output_csv(self, outputfile):
    fout = open(outputfile, 'w')
    fout.write('projectid, essay_length\n')
    #dict_feature = {}
    for i in xrange(self.numprocs):
      for pid, length in iter(self.outq.get, 'STOP'):
        fout.write(','.join([pid, str(length)]) + '\n')
    '''
      #dict_feature.update(\
      #dict((pid, features) for pid, features in iter(self.outq.get, 'STOP')))
    fout.write('\n'.join(','.join(str(feature)\
    for feature in dict_feature[kk]) for kk in pid_train) + '\n')
    fout.write('\n'.join(','.join(str(feature)\
    for feature in dict_feature[kk]) for kk in pid_test) + '\n')
    '''
    fout.close()

if __name__ == '__main__':
  c = CSVWorker(15, 'data/essays.csv', 'features/eassy_len.csv')

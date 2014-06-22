import csv, time
from collections import Counter
from textblob import TextBlob as tb
from mytexttool import *
from readability import getmeasures

tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
tag_dict = {'PRP$': 0.0, 'VBG': 0.0, 'FW': 0.0, 'VBN': 0.0, 'VBP': 0.0, 'WDT': 0.0, 'JJ': 0.0, 'WP': 0.0, 'VBZ': 0.0, 'DT': 0.0, 'RP': 0.0, 'NN': 0.0, 'VBD': 0.0, 'POS': 0.0, 'TO': 0.0, 'PRP': 0.0, 'RB': 0.0, 'NNS': 0.0, 'NNP': 0.0, 'VB': 0.0, 'WRB': 0.0, 'CC': 0.0, 'LS': 0.0, 'PDT': 0.0, 'RBS': 0.0, 'RBR': 0.0, 'CD': 0.0, 'EX': 0.0, 'IN': 0.0, 'WP$': 0.0, 'MD': 0.0, 'NNPS': 0.0, 'JJS': 0.0, 'JJR': 0.0, 'SYM': 0.0, 'UH': 0.0}

class Essay:
  def __init__(self, essay_row):
    self.pid = essay_row[0]
    #self.title = regex_nonprintable.sub(' ', essay_row[2])
    self.essay= regex_nonprintable.sub('', essay_row[5])
    #self.title = essay_row[2]

  def preprocessing(self):
    #self.title_tokens = word_tokenize(remove_punctuation(self.title))
    self.essay_tokens = word_tokenize(remove_punctuation(self.essay))
    
  def features_length(self):
    return (len(self.essay_tokens), )

  def features_pos_tag(self):
    blob = tb(self.essay)
    counts = Counter(tag for word,tag in blob.tags)
    total = sum(counts.values())
    ratio_dict = tag_dict.copy()
    ratio_dict.update(dict((word, float(count)/total) for word,count in counts.items()))
    return tuple(map(lambda k: ratio_dict[k], tag_list))

  def readability_features(self):
    features = []
    n_para = len(re.findall(r'\\n\\n', self.essay))
    measures = getmeasures(sent_detector.tokenize(self.essay)+['']*n_para)
    grades = measures['readability grades']
    features += grades.values()
    sent_info = measures['sentence info']
    features += sent_info.values()
    #features += map(lambda k: sent_info[k],\
    #['characters_per_word', 'syll_per_word', 'words_per_sentence', 'characters','syllables', 'words', 'sentences', 'long_words', 'complex_words'])
    word_usage = measures['word usage']
    features += word_usage.values()
    sent_begin = measures['sentence beginnings']
    features += sent_begin.values()
    return tuple(features)
    
  def features_all(self):
    self.preprocessing()
    return self.features_length()# + self.features_pos_tag()

def get_essay_features():
  essayreader = csv.reader(open('data/essays_fixed.csv'), quotechar='"')
  essayreader.next()
  essay_row = essayreader.next()
  return dict((row[0], Essay(row).features_all()) for row in essayreader)
  '''
  dd = {}
  t0 = time.time()
  for i, row in enumerate(essayreader):
    dd[row[0]] = Essay(row).features_all()
    if (i+1)%1000 == 0:
      print i+1, time.time()-t0
  return dd
  '''

if __name__ == '__main__':
  dd = get_essay_features()

import sys
from datetime import date
import pandas as pd
import cPickle as pickle

from classify import *

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
  data[col+'_scaled'] = (data[col]-min_value)/float(max_value-min_value)
  data.drop(col, axis=1, inplace=True)
  return data

def n_projects_before_by_teacher(data, cutoff_date, teacher):
  mask = (data['teacher_acctid']==teacher) & (data['date_posted']<cutoff_date)
  return mask.sum()
  
def average_performance(group):
  n_train = group['is_exciting_t'].count()
  date_rank = group['date_posted'].rank(ascending=True, method='min').values.astype(np.int32)
  sorted_performance = group.sort('date_posted')[label_t_cols].values
  #avg_performance = []
  #sum_performance
  #for i in xrange(n_train):
  avg_performance = np.nan_to_num(map(lambda rank: sorted_performance[:min(n_train, rank-1)].mean(axis=0), date_rank))
  return pd.DataFrame(avg_performance, columns=['avg_'+l for l in label_t_cols], index=group.index)

def momentum_performance(data, beta):
  inq =  multiprocessing.Queue()
  outq = multiprocessing.Queue()
  def calc():
    for group in iter(inq.get, 'STOP'):
      sorted_perf = group.sort('date_posted')[label_t_cols].values
      momentum_perf = np.zeros(len(label_t_cols))
      last_perf = sorted_perf[0]
      for current_perf in sorted_perf[1:]:f
        last_perf = beta*current_perf + (1-beta)*last_perf
	momentum_perf = np.vstack((momentum_perf, last_perf))
        
  grouped = data[['teacher_acctid', 'date_posted'] + label_t_cols].groupby('teacher_acctid')
  for _, group in grouped:
    inq.put(group)
#def load_features_project():
num_cols = ['total_price_excluding_optional_support','total_price_including_optional_support', 'students_reached']
bin_cols = ['school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'eligible_double_your_impact_match', 'eligible_almost_home_match']
cate_cols = ['fulfillment_labor_materials', 'school_metro','poverty_level', 'teacher_prefix', 'primary_focus_subject', 'primary_focus_area', 'resource_type', 'grade_level']
state_col = ['school_state']
zip_col = ['school_zip']
date_col = ['date_posted']
#id_cols = ['projectid','teacher_acctid','schoolid','school_ncesid']
id_cols = ['teacher_acctid','schoolid','school_ncesid']
drop_cols = ['school_latitude','school_longitude','school_city','school_district','school_county','secondary_focus_subject','secondary_focus_area']
label_cols = ['is_exciting','at_least_1_teacher_referred_donor','fully_funded','at_least_1_green_donation','great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor']
label_t_cols = [l+'_t' for l in label_cols]
#pos_cols = ['pos%d' %(i+1) for i in xrange(36)]
#read_cols = ['read%d' %(i+1) for i in xrange(31)]

t0 = time()
print 'preparing data'


#----------------load initial project & label data-------------#
df_data = pd.read_csv('data/projects_fixed4.csv', index_col='projectid', parse_dates=['date_posted'], dtype={'school_zip': str})
df_new = pd.read_csv('data/outcomes_binary_nomissing.csv', index_col='projectid')
df_data = df_data.merge(df_new, how='left', left_index=True, right_index=True)


#----------------------encode binary features--------------------#
#df_data.school_zip = df_data.school_zip.map(lambda x: x[:3])
df_data = dummy_variables_categorical(df_data, cate_cols+state_col)
df_data = dummy_variables_binary(df_data, bin_cols+label_cols)
#---------------------normalize numerical features--------------------#
df_data = normalize_num(df_data, 'total_price_excluding_optional_support', 10000, 0)
df_data = normalize_num(df_data, 'total_price_including_optional_support', 10000, 0)
df_data = normalize_num(df_data, 'students_reached', 1000, 0)


#-------------number of previsous projects by teacher--------------#
nproj_before_teacher = 'nproj_before_teacher'
num_cols.append(nproj_before_teacher)
df_new = pd.read_csv('features/nproj_before_teacher.csv', index_col='projectid')
df_new = normalize_num(df_new, nproj_before_teacher, 213, 1)
df_data = df_data.merge(df_new, how='left', left_index=True, right_index=True)

#------------------------# of words in essay--------------------#
essay_length = 'essay_length'
num_cols.append(essay_length)
df_new = pd.read_csv('features/essay_len.csv', index_col='projectid')
df_new = normalize_num(df_new, essay_length, 1000, 0)
df_data = df_data.merge(df_new,left_index=True, right_index=True)

#----------------average label of previous projects---------------#
#df_data[nproj_before_teacher] = df_data.groupby('teacher_acctid')['date_posted'].rank(ascending=True, method='min')
#grouped = df_data[['teacher_acctid', 'date_posted'] + label_t_cols].groupby('teacher_acctid')
#df_avg = grouped.apply(average_performance)
#pickle.dump(df_avg, open('df_avg', 'wb'))
#df_avg = pickle.load(open('df_avg', 'rb'))
df_new = pd.read_csv('features/average_label.csv', index_col='projectid')
df_data = df_data.merge(df_new, left_index=True, right_index=True)
'''
#grouped = grouped.transform(lambda x: x.fillna(x.mean()))
#---------------------tfidf---------------------------------------#
t1 = time()
tfidf_projectid, tfidf_vectors = pickle.load(open('tfidf1000.pickle'))
tfidf_cols = ['tfidf%d' % (i) for i in xrange(1000)]
df_tfidf = pd.DataFrame(tfidf_vectors.toarray(), columns = tfidf_cols)
df_tfidf['projectid'] = tfidf_projectid[1:]
df_tfidf.set_index('projectid', inplace=True)
df_data = df_data.merge(df_tfidf, left_index=True, right_index=True)
print 'tfidf loading %d s'%(time()-t1)
'''
'''
#------------# of previous projects by school-------------------------#
nproj_before_school = 'nproj_before_school'
df_data[nproj_before_school] = pickle.load(open(nproj_before_school+'.pickle', 'rb'))
num_cols.append(nproj_before_school)
df_data = normalize_num(df_data, nproj_before_school, 741, 1)
'''
'''
df_pos_read = pd.read_csv('features/essay_pos_readability.csv', index_col='projectid')
df_pos_read[read_cols] = MinMaxScaler().fit_transform(df_pos_read[read_cols].values)
df_data = df_data.merge(df_pos_read[read_cols],left_index=True, right_index=True)
'''
date_cutoff_test = date(2014,1,1)
data_test = df_data[df_data['date_posted'] >= date_cutoff_test]
data_train = df_data[df_data['date_posted'] < date_cutoff_test]

date_cutoff_validate = date(2013,7,1)
data_validate = data_train[data_train['date_posted'] >= date_cutoff_validate]
data_train = data_train[data_train['date_posted'] < date_cutoff_validate]
final_drop_cols = drop_cols+id_cols+date_col+zip_col
#final_drop_cols = drop_cols+id_cols+date_col+state_col
validate = data_validate.drop(final_drop_cols, axis=1)
train = data_train.drop(final_drop_cols, axis=1)
test = data_test.drop(final_drop_cols, axis=1)

n_feature = train.columns.size - 8
Y_train = train[label_t_cols].values.astype(np.int8)
X_train = train.drop(label_t_cols,axis=1).values
Y_validate = validate[label_t_cols].values.astype(np.int8)
X_validate = validate.drop(label_t_cols,axis=1).values
X_test = test.drop(label_t_cols,axis=1).values

print 'in %d s' %(time()-t0)


if __name__ == '__main__':
  print 'training'
  idx_label = int(sys.argv[1])
  cw = float(sys.argv[2])
  alpha = float(sys.argv[3])

  pred = sgd(X_train, Y_train[:,idx_label], X_validate, Y_validate[:,idx_label], X_test, cw, alpha,\
  regression=True)

  submission = pd.DataFrame()
  submission['projectid'] = data_test.index.tolist()
  submission['is_exciting'] = pred
  submission.to_csv('label%dcw%.1f.csv' %(idx_label, cw), index=False)
  #submission.to_csv('label%d.csv' %(idx_label), index=False)

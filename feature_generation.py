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
  
num_cols = ['total_price_excluding_optional_support','total_price_including_optional_support', 'students_reached']
bin_cols = ['school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'eligible_double_your_impact_match', 'eligible_almost_home_match']
cate_cols = ['fulfillment_labor_materials','school_state', 'school_metro','poverty_level', 'teacher_prefix', 'primary_focus_subject', 'primary_focus_area', 'resource_type', 'grade_level']
date_col = ['date_posted']
zip_col = ['school_zip']
id_cols = ['projectid','teacher_acctid','schoolid','school_ncesid']
drop_cols = ['school_latitude','school_longitude','school_city','school_district','school_county','secondary_focus_subject','secondary_focus_area']
label_cols = ['is_exciting','at_least_1_teacher_referred_donor','fully_funded','at_least_1_green_donation','great_chat','three_or_more_non_teacher_referred_donors','one_non_teacher_referred_donor_giving_100_plus','donation_from_thoughtful_donor']

t0 = time()
print 'preparing data'
df_data = pd.read_csv('data/projects_fixed4.csv', parse_dates=['date_posted'], dtype={'school_zip': str})

# count how many projects a teacher posted before current one
nproj_before_teacher = 'nproj_before_teacher'
#df_data[nproj_before_teacher] = df_data.groupby('teacher_acctid')['date_posted'].rank(ascending=True, method='min')
df_data[nproj_before_teacher] = pickle.load(open('features/nproj_before_teacher.pickle', 'rb'))
num_cols.append(nproj_before_teacher)
df_data = normalize_num(df_data, nproj_before_teacher, 228, 1)

'''
nproj_before_school = 'nproj_before_school'
df_data[nproj_before_school] = pickle.load(open(nproj_before_school+'.pickle', 'rb'))
num_cols.append(nproj_before_school)
df_data = normalize_num(df_data, nproj_before_school, 741, 1)
'''

# adding # of words in essay
essay_length = 'essay_length'
num_cols.append(essay_length)
df_data = df_data.merge(pd.read_csv('features/essay_len.csv'))
df_data = normalize_num(df_data, essay_length, 1000, 0)

df_data = dummy_variables_categorical(df_data, cate_cols)
df_data = dummy_variables_binary(df_data, bin_cols)
df_data = normalize_num(df_data, 'total_price_excluding_optional_support', 10000, 0)
df_data = normalize_num(df_data, 'total_price_including_optional_support', 10000, 0)
df_data = normalize_num(df_data, 'students_reached', 1000, 0)

date_cutoff_test = date(2014,1,1)
data_test = df_data[df_data['date_posted'] >= date_cutoff_test]
data_train = df_data[df_data['date_posted'] < date_cutoff_test]

df_labels = pd.read_csv('data/outcomes_binary_nomissing.csv')
df_labels = dummy_variables_binary(df_labels, label_cols)
data_train = data_train.merge(df_labels)

date_cutoff_validate = date(2013,7,1)
data_validate = data_train[data_train['date_posted'] >= date_cutoff_validate]
data_train = data_train[data_train['date_posted'] < date_cutoff_validate]
final_drop_cols = drop_cols+id_cols+zip_col+date_col
validate = data_validate.drop(final_drop_cols, axis=1)
train = data_train.drop(final_drop_cols, axis=1)
test = data_test.drop(final_drop_cols, axis=1)

n_feature = train.columns.size - 8
X_train = train.values[:,:n_feature]
X_validate = validate.values[:,:n_feature]
Y_train = train.values[:,n_feature:].astype(np.int8)
Y_validate = validate.values[:,n_feature:].astype(np.int8)
X_test = test.values

print 'in %d s' %(time()-t0)

if __name__ == '__main__':
  print 'training'
  idx_label = int(sys.argv[1])
  cw = float(sys.argv[2])
  pred = sgd(X_train, Y_train[:,idx_label], X_validate, Y_validate[:,idx_label], X_test, cw)
  submission = pd.DataFrame(data_test['projectid'])
  #submission['label%d' %(idx_label)] = pred
  submission['target'] = pred
  submission.to_csv('label%dcw%.1f.csv' %(idx_label, cw), index=False)
  #submission.to_csv('label%d.csv' %(idx_label), index=False)

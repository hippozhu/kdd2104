import numpy as np
from collections import namedtuple
from datetime import date

project_fields = ['project_id', 'teacher_id', 'school_id', 'subject', 'area', 'resourse', 'poverty', 'grade', 'labor', 'price1', 'price2', 'reached', 'match1', 'match2', 'date']
project_fields_index = [0, 1, 2, 21, 22] + range(25, 34)
project_date_index = 34
Project = namedtuple('Project', project_fields)
    
def load_projects(project_file):
  dict_project_train = {}
  dict_project_test = {}
  fin = open(project_file)
  project_feature_names = fin.next().strip().split(',')
  for line in fin:
    contents = line.strip().split(',')
    project = Project._make(map(lambda i: contents[i], project_fields_index) + [date(*map(lambda x: int(x), contents[project_date_index].split('-')))])
    if project.date > date(2013, 12, 31):
      dict_project_test[contents[0]] = project
    else:
      dict_project_train[contents[0]] = project
  fin.close()
  return dict_project_train, dict_project_test

school_fields = ['school_id', 'state', 'zip', 'charter', 'magnet', 'year_round', 'nlns', 'kipp', 'chater_promise'] 
school_fields_index = [2,7,8] + range(12, 18)
School = namedtuple('School', school_fields)

def load_schools(school_file):
  dict_school = {}
  fin = open(school_file)
  for line in fin:
    contents = line.strip().split(',')
    dict_school[contents[0]] = School._make(contents)
  fin.close()
  return dict_school

teacher_fields = ['teacher_id', 'prefix', 'america', 'ny']
teacher_fields_index = [1, 18, 19, 20]
Teacher = namedtuple('Teacher', teacher_fields)

def load_teachers(school_file):
  dict_school = {}
  fin = open(school_file)
  for line in fin:
    contents = line.strip().split(',')
    dict_school[contents[0]] = Teacher._make(contents)
  fin.close()
  return dict_school

outcome_fields = ['project_id', 'exciting', 'teacher_referred', 'funded', 'green', 'chat', 'non_teacher_referred', 'plus', 'thoughtful']
Outcome = namedtuple('Outcome', outcome_fields)

def load_outcomes(outcome_file):
  dict_outcome = {}
  fin = open(outcome_file)
  fin.next()
  for line in fin:
    contents = line.strip().split(',')
    dict_outcome[contents[0]] = Outcome._make(contents)
  fin.close()
  return dict_outcome


def load_entity(filename, entity, skip_header):
  dict = {}
  fin = open(filename)
  if skip_header:
    fin.next()
  for line in fin:
    contents = line.strip().split(',')
    dict[contents[0]] = entity._make(contents)
  fin.close()
  return dict

def toArray(outcome):
  return np.array(map(lambda label: 1 if label=='t' else 0, outcome[1:]), dtype=np.int32)

def projects_by_teacher(projects_train, teachers):
  dict_projects = dict((tid, []) for tid in teachers.keys())
  for project in projects_train.values():
    dict_projects[project.teacher_id].append(project.project_id)
  for pids in dict_projects.values():
    pids.sort(key = lambda pid: projects_train[pid].date)
  return dict_projects 

def projects_by_school(projects_train, schools):
  dict_projects = dict((sid, []) for sid in schools.keys())
  for project in projects_train.values():
    dict_projects[project.school_id].append(project.project_id)
  for pids in dict_projects.values():
    pids.sort(key = lambda pid: projects_train[pid].date)
  return dict_projects 

'''
pid_train, labels = load_outcome('data/outcomes_binary_nomissing.csv')
pid_test = [line.strip() for line in open('test_projects')]
projects = load_projects('data/projects_fixed4.csv')
projects_train = map(lambda pid: projects[pid], pid_train)
projects_test = map(lambda pid: projects[pid], pid_test)

projects_train = dict(map(lambda pid: (pid, projects[pid]), pid_train))
projects_test = dict(map(lambda pid: (pid, projects[pid]), pid_test))

schools = load_schools('data/schools.csv')
teachers =  load_teachers('data/teachers.csv')
outcomes = load_entity('data/outcomes_binary_nomissing.csv', Outcome, True)
'''

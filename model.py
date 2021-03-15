#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
This is Project 2 from Group 4 of DS-21-1 - March 2021
Group 4: Christian Steck, Christoph Michel, Kay Delventhal

We have analysed a Kickstarter data set, in order to predict,
the success of a compain by given features:
'cat_main'           - main categories from Kickstarter
'cat_sub'            - sub categories from Kickstarter
'name_wc'            - word count of campain name
'slug_wc'            - # of tags
'launch_to_deadline' - total time for campaign
'usd_goal'           - target amount of money in USD

With the created model we create we can predict the chances of success for 
campaign under consideration of given features above.
"""

__author__ = "Gruppe 4"
__license__ = "MIT"

import glob
import os
import string
import numpy as np
import pandas as pd
import pickle
from time import time
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, \
                            precision_score, f1_score, fbeta_score, make_scorer, classification_report

from sklearn.linear_model import LogisticRegression

RSEED = 42
CHUNCK = 0.99
TSIZE = 0.3

cats = ['cat_main', 'cat_sub'] # best without 'staff_pick' !
nums = ['name_wc', 'slug_wc', 'launch_to_deadline', 'usd_goal'] # best without 'staff_pick' !
targ = ['state']

def headline(header):
    '''print helper'''
    print('='*80) 
    print(header) 
    print('='*80) 

def subline(line):
    '''print helper'''
    print(line) 
    print('-'*80) 

def save_csv(data,ftemp):
    '''save pandas data as csv file'''
    data.to_csv(ftemp)

def read_csv(ftemp):
    '''read pandas data from csv file'''
    return pd.read_csv(ftemp)

def import_data():
    '''import and features engineer project data'''
    # load .csv (56) files 
    subline('load csv files')
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('data', '*.csv'))))
    
    # the duplicate ids seem to be actual duplicates. Drop them (drop last)
    print('drop suplicates ...')
    df=df.drop_duplicates(subset='id')

    # change state 'successful' and 'failed' to 1 and 0, drop rest.
    print('change "state" ...')
    df = df[(df["state"] == "successful") | (df["state"] == "failed")]
    booleans = {'successful': 1, 'failed': 0}
    df['state'] = df['state'].map(booleans)

    # 'goal' is not necessarily in USD. We see in idx 22, that pledged > goal, 
    # but usd_pledged < goal and static_usd_rate < 1. We get the comparable 
    # goal in USD if we multiply it by the static_usd_rate
    print('change "usd_goal" ...')
    df['usd_goal'] = df.goal*df.static_usd_rate

    # make new variable containing the duration of the campaign
    # time setting for duration (of project? until cut-off)
    print('create "launch_to_deadline" ...')
    df["launch_to_deadline"] = df["deadline"] - df["launched_at"]

    # set new variable for proportion of goal reached
    print('create "pledge_to_goal" ...')
    df["pledge_to_goal"] = df["usd_pledged"] / df["usd_goal"]

    # count words of project description, name of project and number of tags
    # get rid of standalone punctuation, to avoid '-' beeing counted as a word
    print('create "blurb_wc" ...')
    df['blurb_wc'] = df.blurb.str.replace(f' [{string.punctuation}]', '').str.split().str.len()
    # get rid of standalone punctuation, to avoid '-' beeing counted as a word
    print('create "name_wc" ...')
    df['name_wc'] = df.name.str.replace(f' [{string.punctuation}]', '').str.split().str.len() 
    print('create "slug_wc" ...')
    df['slug_wc']= df.slug.str.split('-').str.len()

    # get the categories, main-categories, sub-categories and names from the 'category-column'-dictionaries
    print('create "categories" ...')
    df['categories'] = df['category'].map(eval).apply(lambda x: x['slug'].split('/'))
    print('create "cat_name" ...')
    df['cat_name'] = df['category'].map(eval).apply(lambda x: x['name']) 
    print('create "cat_main" ...')
    df['cat_main'] = df['categories'].apply(lambda x: x[0])
    print('create "cat_sub" ...')
    df['cat_sub'] = df['categories'].apply(lambda x: x[1] if len(x)==2  else '0')

    # cut projects with high or low goals
    print('reduce data set (based usd_goal) ...')
    data = df[(df["usd_goal"]<=1000000.0) & (df["usd_goal"] >= 10)]

    # casting to 0 and 1
    print('convert "staff_pick" ...')
    data.staff_pick = data.staff_pick.astype(int)

    # save data
    subline('save data')
    fname = 'data/cleaned_data.csv'
    save_csv(data, fname)
    print('data saved:', fname)

    return data

def data_split(work,savedata='data/saved.csv'):
    '''split pandas data frame into train, test and saved data'''
    if CHUNCK != 1:
        save, temp = train_test_split(work, test_size=CHUNCK, random_state=RSEED)
    else:
        temp = data.copy()
    print('data shape   :', work.shape)
    print('save shape   :', save.shape)
    print('temp shape   :', temp.shape)
    features = temp[cats+nums]
    tragets = temp[targ]

    # save & load data
    fname = savedata
    save_csv(save, fname)

    X_train, X_test, y_train, y_test = train_test_split(features, tragets, 
                                                test_size = TSIZE, random_state = RSEED)
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape :', X_test.shape)
    print('y_test shape :', y_test.shape)
    return X_train, X_test, y_train, y_test

def pipeline(X_train, y_train, cv=5):
    '''pipeline steup for model craetion - baaed on: LogisticRegression()'''
    subline('create pipeline')

    # Pipline for numerical features
    print('setup: num_pipeline ...')
    num_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    print('setup: cat_pipeline ...')
    cat_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='missing')),
        ('1hot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Complete pipeline for numerical and categorical features
    print('setup: preprocessor ...')
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, nums),
        ('cat', cat_pipeline, cats)
    ])

    # Building a full pipeline with our preprocessor and a LogisticRegression Classifier
    print('setup: Pipeline ...')
    pipe_model = Pipeline([('prep', preprocessor),
                        ('model', LogisticRegression(max_iter=1000,random_state=RSEED, 
                        n_jobs = -1))])

    # cross_val_predict expects an estimator (model), X, y and nr of cv-splits (cv)
    print('cross_val_predict ...')
    y_train_predicted = cross_val_predict(pipe_model, X_train, np.ravel(y_train), cv=cv)

    # fit model
    print('fit model ...')
    pipe_model.fit(X_train, np.ravel(y_train))
    return pipe_model, preprocessor

def report(y, y_predicted,):
    '''print y versus y_predict analyses'''
    # Calculating the accuracy, recall and precision for the test set with the optimized model    
    print('\n')
    print("Accuracy   : {:.5f}".format(accuracy_score(y, y_predicted)))
    print("Recall     : {:.5f}".format(recall_score(y, y_predicted)))
    print("Precision  : {:.5f}".format(precision_score(y, y_predicted)))

    print('\n')
    print('confusion_matrix')
    print(str(confusion_matrix(y, y_predicted)))
    print('\n')
    print('classification_report')
    print(classification_report(y, y_predicted))

def prediction(model, X_test):
    '''predict y from X by given model'''
    return model.predict(X_test)

def save_model(model, fname):
    '''save model to file'''
    pickle.dump(model, open(fname, 'wb'))

def read_model(fname):
    '''read model from file'''
    return pickle.load(open(fname, 'rb'))


if __name__ == "__main__":
    '''
    this is the main function - use can chose between options:
    option 1: will use project data to train a model
    option 2: will use the trained model to predict from given data
    '''

    choice = input('1) create model -  2) predict y: ')

    if '1' in choice:
        headline('import and engineer data')
        data = import_data()

        headline('split data')
        #CHUNCK = float(input('enter CHUNCK size (0,1):'))
        X_train, X_test, y_train, y_test = data_split(data)

        headline('create and train pipeline:')
        print(f'categories: {cats}')
        print(f'numericals: {nums}')
        print(f'X data    : {X_train.shape}')
        print(f'y data    : {y_train.shape}')
        print(f'CHUNCK    : {CHUNCK}')
        print(f'TSIZE     : {TSIZE}')
        print(f'RSEED     : {RSEED}')
        model, preprocessor = pipeline(X_train, y_train, cv=5)

        headline('predict train data')
        y_train_pred = prediction(model, X_train)
        report(y_train, y_train_pred)

        headline('predict test data')
        y_test_pred = prediction(model, X_test)
        report(y_test, y_test_pred)

        headline('save model')
        fname = 'data/model.pkl'
        save_model(model, fname)
        print('model saved:', fname)
        fname = 'data/preprocessor.pkl'
        save_model(preprocessor, fname)
        print('preprocessor saved:', fname)
        print('='*80)

    elif '2' in choice:

        headline('load model')
        fname = 'data/model.pkl'
        model = read_model(fname)
        print('loaded model:', fname)

        headline('input data to predict')
        #ftemp = input('read X csv file: ')
        ftemp = 'data/saved.csv'
        data = read_csv(ftemp)
        X_train, X_test, y_train, y_test = data_split(data,savedata='data/temp.csv')
        print('loaded data:', ftemp)
        
        headline('predict model')
        y_pred = prediction(model, X_train)
        
        headline('predict report')
        report(y_pred, y_train)

        headline('save model')
        fname = 'data/result.csv'
        save_csv(pd.DataFrame(y_pred, columns=['target']),fname)
        print('model saved:', fname)

# EOF
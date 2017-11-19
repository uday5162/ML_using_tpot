from tpot import TPOTClassifier
from sklearn.cross_validation import train_test_split
import pandas as pd 
import numpy as np


#load the data
train=pd.read_csv('train_data.csv')

feature_names = [x for x in train.columns if x not in ['connection_id','target']]
target = train['target']

X_train, X_valid, y_train, y_valid = train_test_split(train, target, train_size = 0.7, stratify = target, random_state = 2017)

tpot = TPOTClassifier(generations=5, verbosity=2)

tpot.fit(X_train[feature_names], y_train)

tpot.score(X_valid[feature_names],y_train)

tpot.export('pipeline.py')
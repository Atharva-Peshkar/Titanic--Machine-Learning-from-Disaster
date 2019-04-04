# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:22:22 2019

@author: Dreams
"""

#importing the libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing 


#Get the training and test data
df = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#preparing the data
data = df[['PassengerId','Pclass','Sex','Age','Survived']].fillna(0)
transformer = preprocessing.LabelEncoder()
data['Sex'] = transformer.fit_transform(data['Sex'])
X = data[['Pclass','Sex','Age']]
y = np.ravel(data[['Survived']])

test_input=test_data[['Pclass','Sex','Age']].fillna(0)
transformer = preprocessing.LabelEncoder()
test_input['Sex'] = transformer.fit_transform(test_input['Sex'])


#fitting the dataset
predictor = GaussianNB()
predictor.fit(X,y)

#predicting the values
y_pred = predictor.predict(test_input)
final = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':y_pred})
final.to_csv('titanic submission.csv',encoding='utf-8',index=False)

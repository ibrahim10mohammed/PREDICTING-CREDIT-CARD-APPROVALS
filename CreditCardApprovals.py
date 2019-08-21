# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 19:31:43 2019

@author: ibrahim
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


cc_apps = pd.read_csv('cc_approvals.csv')

print(cc_apps.head(5))

cc_apps_description = cc_apps.describe()
print(cc_apps_description)

cc_apps_info = cc_apps.info
print(cc_apps_info)

print("\n")

print(cc_apps.tail(17))

# Replace the '?'s with NaN
cc_apps = cc_apps.replace('?',np.nan)

print(cc_apps.tail(17))

cc_apps.fillna(cc_apps.mean(), inplace=True)

pd.isna(cc_apps)
cc_apps.tail(17)

# Iterate over each column of cc_apps
cc_apps=cc_apps.fillna(method='ffill')
# Count the number of NaNs in the dataset and print the counts to verify
# ... YOUR CODE FOR TASK 5 ...
print(cc_apps.count())
cc_apps.tail(20)

le=LabelEncoder()
# Iterate over all the values of each column and extract their dtypes
for col in cc_apps:
    # Compare if the dtype is object
    if cc_apps[col].dtype=='object':
    # Use LabelEncoder to do the numeric transformation
         cc_apps[col]=le.fit_transform(cc_apps[col])
cc_apps.head()

cc_apps = cc_apps.drop(columns=['01', '00202'], axis=1)
cc_apps = cc_apps.values

# Segregate features and labels into separate variables
rescaledX,y = cc_apps[:,0:13] , cc_apps[:,13]

X_train, X_test, y_train, y_test = train_test_split(rescaledX,y,test_size=0.33,random_state=42)

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ",logreg.score(X_test,y_test))

# Print the confusion matrix of the logreg model

print(confusion_matrix(y_test,y_pred))




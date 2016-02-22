import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import numpy as np
import math
##
##df = pd.read_csv('C:/temp/machine learning/courseraYa/gbm-data.csv', header=0)
##X = df.ix[:, df.columns != 'Activity']
##X = X.values
##y = df['Activity']
##y = y.values
data = pd.read_csv("C:/temp/machine learning/courseraYa/gbm-data.csv").values

y= data[:, 0]

X= data[:, 1:]
#split into train test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

#train
clf = RandomForestClassifier(n_estimators=36,  random_state=241)
clf.fit(X_train, y_train)

#verify log loss
rfc_loss = log_loss(y_test,clf.predict_proba(X_test))
print(rfc_loss)



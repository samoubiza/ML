import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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
clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=0.2)
clf.fit(X_train, y_train)

#verify log loss


loss_on_test = []

for i, pred1 in enumerate(clf.staged_decision_function(X_test)):
##    print(i)
##    print(pred1)
##    print(y_test)
    x = log_loss(y_test, 1.0/(1.0+np.exp(-pred1)))
##    print(x)
    loss_on_test.append(x)

grd2 = clf.staged_predict_proba(X_test)

loss_on_test_proba = []

for i, pred2 in enumerate(grd2):

    loss_on_test_proba.append(log_loss(y_test, pred2))

print(min(loss_on_test))
print(min(loss_on_test_proba))
print(loss_on_test_proba.index(min(loss_on_test_proba)))


loss_on_train = []

for i, pred3 in enumerate(clf.staged_decision_function(X_train)):
##    print(i)
##    print(pred1)
##    print(y_test)
    x = log_loss(y_train, 1.0/(1.0+np.exp(-pred3)))
##    print(x)
    loss_on_train.append(x)


plt.figure()
plt.plot(loss_on_train, 'g', linewidth=2)
plt.plot(loss_on_test, 'r', linewidth=2)
plt.legend(["train", "test"])
plt.show()

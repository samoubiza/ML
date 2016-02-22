import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import pandas as pd
data_test = pd.read_csv('C:/temp/machine learning/courseraYa/perceptron-test.csv', header=0)
data_train = pd.read_csv('C:/temp/machine learning/courseraYa/perceptron-train.csv', header=0)
y_train = data_train.iloc[:,0] #classes / target values
X_train = data_train.iloc[:,1:] #feaches

y_test = data_test.iloc[:,0] #classes / target values
X_test = data_test.iloc[:,1:] #feaches

clf = Perceptron(random_state=241, shuffle = True)
clf.fit(X_train, y_train)
#predictions = clf.predict(X_test)
acur = clf.score(X_test,y_test)
print(acur)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_scaled = Perceptron(random_state=241, shuffle = True)
clf_scaled.fit(X_train_scaled, y_train)
#predictions = clf.predict(X_test)
acur_scaled = clf_scaled.score(X_test_scaled,y_test)
print(acur_scaled)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation  import KFold, cross_val_score
import pandas as pd
import re
df = pd.read_csv('C:/temp/machine learning/courseraYa/abalone.csv', header=0)
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
##print(df.head(3))
##header = list(df.columns.values)
##print(df.shape)
##print(header)
l = len(df)
y = df['Rings']
targets = y.values
X = df.ix[:, df.columns != 'Rings']
features = X.values

d={}

for i in range(1,51):

    clf = RandomForestRegressor(n_estimators=i,random_state=1)
    kf = KFold(l, n_folds=5, shuffle = True, random_state = 1)

    for train_id, test_id in kf:

        X_train, X_test = X.iloc[train_id], X.iloc[test_id]

        y_train , y_test = y.iloc[train_id], y.iloc[test_id]

        clf.fit(X_train,y_train)
        y_predict = clf.predict(X_test)

        scores = r2_score(y_test, y_predict)
        s =round(scores,2)

        if i in d.keys():
            d[i].append(s)
        else:
            d[i]=[s]



for key in d:
    m = sum(d[key])/len(d[key])
    d[key].append(m)
d_new={}
for key in d:
    d_new[key] = d[key][5]

print(sorted(d_new.items() , key = lambda x: x[1]))

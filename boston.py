from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation  import  KFold,  cross_val_score
boston = load_boston()
#boston.data, boston.target
X = scale(boston.data)
y = boston.target
d={}
l = len(boston.target)
#print(l)
for i in np.linspace(1.0,10.0, num=200):
    #print(i)
    neigh = KNeighborsRegressor(n_neighbors=5,  metric= 'minkowski',weights='distance', p = i)
    kf = KFold(l, n_folds=5, shuffle = True, random_state = 42)

    for train_id, test_id in kf:

        X_train, X_test = X[train_id], X[test_id]

        y_train , y_test = y[train_id], y[test_id]
        neigh.fit(X_train, y_train)
        scores = cross_val_score(neigh, X_train,y_train, scoring = 'mean_squared_error', cv =5)
        #print(scores)
        s =scores.mean()
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




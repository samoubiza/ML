from sklearn.preprocessing import scale
import pandas as pd
data = pd.read_csv( 'C:/temp/machine learning/courseraYa/wine.data' , header = None)
y = data.iloc[:,0] #classes / target values
X = data.iloc[:,1:] #feaches
#X = scale(X)
l = len(data)
#print(l)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation  import KFold, cross_val_score


d={}

for i in range(1,51):
    #print(d)
    neigh = KNeighborsClassifier(n_neighbors = i)
    kf = KFold(l, n_folds=5, shuffle = True, random_state = 42)

    for train_id, test_id in kf:

        X_train, X_test = X.iloc[train_id], X.iloc[test_id]

        y_train , y_test = y[train_id], y[test_id]
        #X_train_scaled = scale(X_train)
        #X_test_scaled = scale(X_test)
        neigh.fit(X_train,y_train)
        #print(i)
        scores = cross_val_score(neigh, X_train, y_train, scoring = 'accuracy', cv =5)
        #print(scores)
        s =scores.mean()
        #print(s)
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



#the same result
##
##d={}
##
##
##kf = KFold(l, n_folds=5, shuffle = True, random_state = 42)
##
##for train_id, test_id in kf:
##    X_train, X_test = X.iloc[train_id], X.iloc[test_id]
##    y_train , y_test = y.iloc[train_id], y[test_id]
##    for i in range(1,25):
##    #print(d)
##        neigh = KNeighborsClassifier(n_neighbors = i)
##        neigh.fit(X_train,y_train)
##        #print(i)
##        scores = cross_val_score(neigh, X_test, y_test, cv =5)
##        s =scores.mean()
##        #print(s)
##        if i in d.keys():
##            d[i].append(s)
##        else:
##            d[i]=[s]
##
##
##
##for key in d:
##    m = sum(d[key])/len(d[key])
##    d[key].append(m)
##d_new={}
##for key in d:
##    d_new[key] = d[key][5]rfr
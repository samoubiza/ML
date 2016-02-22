import pandas as pd
df = pd.read_csv('C:/temp/machine learning/courseraYa/classification.csv', header=0)
x = df.values
##x = data[:,1:]
##y = data[:,0]
tp = 0
fp =0
tn = 0
fn = 0

for i in range(len(x)):
    if x[i][1]==1 and x[i][0]==1:
        tp+=1
    elif x[i][1]==1 and x[i][0]==0:
        fp+=1
    elif x[i][1]==0 and x[i][0]==0:
        tn+=1
    elif x[i][1]==0 and x[i][0]==1:
        fn+=1

print(tp, fp, fn, tn)
y_pred = x[:,1]
y_true = x[:,0]


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print('%.2f' % accuracy)
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)
print('%.2f' % precision)
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)
print('%.2f' % recall)
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
print('%.2f' % f1)
import pandas as pd
df = pd.read_csv('C:/temp/machine learning/courseraYa/scores.csv', header=0)
x = df.values

y_true = x[:,0]
score_logreg = x[:,1]
score_svm = x[:,2]
score_knn= x[:,3]
score_tree = x[:,4]

from sklearn.metrics import roc_auc_score
roc_logreg=roc_auc_score(y_true, score_logreg)
roc_svm=roc_auc_score(y_true, score_svm)
roc_knn=roc_auc_score(y_true, score_knn)
roc_tree=roc_auc_score(y_true, score_tree)

print(roc_knn, roc_logreg,roc_svm, roc_tree)

from sklearn.metrics import precision_recall_curve
max_precision_reg = 0
precision_reg, recall_reg, thresholds_reg = precision_recall_curve(y_true, score_logreg)


for i in range(len(precision_reg)):
    if recall_reg[i]>=0.7:
        if max_precision_reg < precision_reg[i]:
            max_precision_reg = precision_reg[i]

print('max_precision_reg:', max_precision_reg)


max_precision_svm = 0
precision_svm, recall_svm, thresholds_svm = precision_recall_curve(y_true, score_svm)


for i in range(len(precision_svm)):
    if recall_svm[i]>=0.7:
        if max_precision_svm < precision_svm[i]:
            max_precision_svm = precision_svm[i]

print('max_precision_svm:', max_precision_svm)

max_precision_knn = 0
precision_knn, recall_knn, thresholds_knn = precision_recall_curve(y_true, score_knn)


for i in range(len(precision_knn)):
    if recall_knn[i]>=0.7:
        if max_precision_knn < precision_knn[i]:
            max_precision_knn = precision_knn[i]

print('max_precision_knn:', max_precision_knn)

max_precision_tree = 0
precision_tree, recall_tree, thresholds_tree = precision_recall_curve(y_true, score_tree)


for i in range(len(precision_tree)):
    if recall_tree[i]>=0.7:
        if max_precision_tree < precision_tree[i]:
            max_precision_tree = precision_tree[i]

print('max_precision_tree:', max_precision_tree)


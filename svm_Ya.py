from sklearn.svm import SVC
import pandas as pd
df = pd.read_csv('C:/temp/machine learning/courseraYa/svm-data.csv', header=None)
data = df.values
features_train = data[:,1:]
labels_train = data[:,0]
print(labels_train)
print(features_train)

clf = SVC(kernel='linear', C = 10000.0, random_state=241)
clf.fit(features_train, labels_train)
print(clf.support_) #indexes point of support vector
print(clf.coef_)
##pred = clf.predict(features_test)
##accuracy = clf.score(features_test, labels_test)
##print(accuracy)



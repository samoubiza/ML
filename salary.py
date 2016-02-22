import pandas as pd
import re
df_test = pd.read_csv('C:/temp/machine learning/courseraYa/salary-test-mini.csv', header=0)
df = pd.read_csv('C:/temp/machine learning/courseraYa/salary-train.csv', header=0)
unique_time = pd.unique(df.ContractTime.ravel())
unique_loc = pd.unique(df.LocationNormalized.ravel())
df['LocationNormalized'].fillna('nan', inplace=True)
df['ContractTime'].fillna('nan', inplace=True)
df_test['LocationNormalized'].fillna('nan', inplace=True)
df_test['ContractTime'].fillna('nan', inplace=True)
X_test = df_test.values
X_train = df.values
y = df['SalaryNormalized'].values
##x = data[:,1:]
##y = data[:,0]
header = list(df.columns.values)
#print(df.shape)
l =len(df.index)
#print(X_train[0,0])
for i in range(l):
    X_train[i,0] = re.sub('[^a-zA-Z0-9]', ' ', X_train[i,0].lower())

print(1)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5)
desc_vector_train = vectorizer.fit_transform(X_train[:,0])
desc_vector_test = vectorizer.transform(X_test[:,0])
print(2)
print(desc_vector_train.shape)
from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer()
X_train_categ = enc.fit_transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
print(3)
print(X_train_categ.shape)
from scipy.sparse import coo_matrix, hstack

h_train = hstack([desc_vector_train, X_train_categ])#.toarray()
h_test = hstack([desc_vector_test, X_test_categ])#.toarray()
print(4)
print(h_train.shape)
from sklearn import linear_model
clf = linear_model.Ridge(alpha = 1.0)
ridge = clf.fit(h_train,y)
print(5)
salary = clf.predict(h_test)
print(salary)
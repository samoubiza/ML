
def main():
    pass

if __name__ == '__main__':
    main()
import csv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('C:/temp/python/coursera/train.csv', header=0)


df = data[ ['Survived','Pclass', 'Fare', 'Age', 'Sex'] ]
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df_new = df.dropna(subset = ['Survived','Pclass', 'Fare', 'Age', 'Gender'])
df_new = df_new.drop(['Sex'], axis=1)
train_data = df_new.values

##
##forest = RandomForestClassifier(random_state=241)
##forest = forest.fit(train_data[0::,1::],train_data[0::,0])
##print(forest)
##score = forest.score(train_data[0::,1::],train_data[0::,0])
##print(score)
##selector = SelectKBest(f_classif)
##selector.fit(df_new[0::,1::],df_new[0::,0])
##
##scores = -np.log10(selector.pvalues_)
##
##plt.bar(range(len(df_new)), scores)
##plt.xticks(range(len(df_new)), df_new[0::,1::], rotation='vertical')
##plt.show()

from sklearn.tree import DecisionTreeClassifier
X = train_data[0::,1::]
y = train_data[0::,0]
clf = DecisionTreeClassifier()
clf.fit(X, y)
importances = clf.feature_importances_
print(importances)
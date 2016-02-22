import pandas as pd
df = pd.read_csv('C:/temp/machine learning/courseraYa/close_prices.csv', header=0)
header = list(df.columns.values)
#print(header)
data= df.ix[:, df.columns != 'date']
##print(df.shape)
##print(df.head(4))
#print(data.head(4))
X = data.values
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)
ratio = list(pca.explained_variance_ratio_)
i = ratio.index(max(ratio))
component = list(pca.components_)
best_compon = list(component[i])
print(len(best_compon))
best_i = best_compon.index(max(best_compon))
print(header[best_i+1])
##
##X1 = X[:, 1]
##Dow_Jones = pd.read_csv('C:/temp/machine learning/courseraYa/djia_index.csv', header=0)
##header = list(Dow_Jones.columns.values)
##idx =Dow_Jones[['^DJI']]
##idx = idx.values
##print(X1.shape)
##idx = idx[:,0]
##print(idx.shape)
##corr = np.corrcoef(X1, idx)
##print(corr)
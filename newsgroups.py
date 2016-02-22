from sklearn import datasets
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation  import KFold
from sklearn.svm import SVC
import numpy as np

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

X = newsgroups.data
y = newsgroups.target


clf = SVC(kernel='linear', random_state=241) # create ckassificator
grid = {'C':  [10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3),10**4, 10**5]}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241) #divide X for 5 parts

vectorizer = TfidfVectorizer() #create vectorizer

X_train = vectorizer.fit_transform(X) #convert words to vectors

gs_clf = GridSearchCV(clf, grid, scoring='roc_auc' , cv=cv) #create GridSearchCV

gs_clf.fit(X_train,y) #training


cf = gs_clf.best_estimator_
dens_f = cf.coef_.todense()

sort_f = np.absolute(np.asarray(dens_f)).reshape(-1)

bst_f = np.argsort(sort_f)[-10:]
bst_w = []
for i in bst_f:
    bst_w.append(vectorizer.get_feature_names()[i])

print(sorted(bst_w))
##
##TfidfVectorizer().get_feature_names()
##get_feature_names()[11098]
##argsort(absolute(asarray(clf.best_estimator_.coef_.todense()).reshape(-1)))
##
###a****,a****,b****,g****,k****,m****,n****,r****,s****,s****







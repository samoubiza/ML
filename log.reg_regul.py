##import numpy as np
##from sklearn.metrics import roc_auc_score
##y_true = np.array([0, 0, 1, 1])
##y_scores = np.array([0.1, 0.4, 0.35, 0.8])
##a =roc_auc_score(y_true, y_scores)
##print(a)
import math
import pandas as pd
df = pd.read_csv('C:/temp/machine learning/courseraYa/data-logistic.csv', header=None)
data = df.values
x = data[:,1:]
y = data[:,0]
l =len(y)
##def gradient_descent(x, y, iters, alpha):
##    costs = []
##    m = y.size # number of data points
##    theta = np.random.rand(2) # random start
##    history = [theta] # to store all thetas
##    preds = []
##    for i in range(iters):
##        pred = np.dot(x, theta)
##        error = pred - y
##        cost = np.sum(error ** 2) / (2 * m)
##        costs.append(cost)
##
##        if i % 25 == 0: preds.append(pred)
##
##        gradient = x.T.dot(error)/m
##        theta = theta - alpha * gradient  # update
##        history.append(theta)
##
##    return history, costs, preds
w1 = 0
w2 = 0
optimal = [10000000 , 1, 1]
n=1
k = 0.1
C=10
while n <=10000:
    w2_his = w2
    w1_his = w1
    cost = 0
    for i in range(len(y)):
        cost += (math.log(1+math.exp(-y[i]*(w1*x[i][0]+w2*x[i][1]))))

    cost = cost/l+k*C*(w1**2+w2**2)
    #print(n)
    #print(cost)
    n+=1
    if optimal[0]>cost:
        optimal = [cost, w1, w2]

    s1,s2 = 0,0
    for i in range(len(y)):
        s1 += y[i]*x[i][0]*(1-1/(1+math.exp(-y[i]*(w1*x[i][0]+w2*x[i][1]))))
        s2 += y[i]*x[i][1]*(1-1/(1+math.exp(-y[i]*(w1*x[i][0]+w2*x[i][1]))))
    #print(s1,s2)
    w1 = w1 + (k/l)*s1-k*C*w1
    w2 = w2 + (k/l)*s2 - k*C*w2
    #print(w1,w2)
    d = math.sqrt((w1-w1_his)**2 + (w2-w2_his)**2)

##    if d < 10**(-5):
##        break
print(optimal)
print(n)
##print(d)
prob = []
for i in range(len(y)):
    prob.append(1/(1+math.exp(-x[i][0]*optimal[1]-x[i][1]*optimal[2])))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y, prob)
print(auc)
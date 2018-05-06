# Thomas Stolp
#
# Bsc-eindwerk
# machine learning in waterveiligheid: voorspellingen van de stijghoogte in dijken

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

h = np.loadtxt('../Data/h')
P = np.loadtxt('../Data/P')
E = np.loadtxt('../Data/E')

print("size of h = ", np.shape(h))
print("size of P = ", np.shape(P))
print("size of E = ", np.shape(E))

a = np.shape(h)
N = 24*365 # uren in één jaar 

print("N = ", N)
# Output values, define a training set and testing set
h_train = h[0:N][:]
h_test = h[N:a[0]][:]

# length of the sets
n_train = len(h_train)
n_test = len(h_test)

print("n_train  = ", n_train)
print("n_test = ", n_test)

def features(P, E, n_train, n_test, lag=4):
    """
    Function that returns an array with features for both the training and testing sets
    
    P: Array met neerslag in mm/uur
    lag: P_i-lag met lag in uren
    E: Array met verdamping in mm/uur
    n_train: Size van trainingset
    n_test: Size van testset
    """
    size = lag + 2
    
    q_train = np.zeros((n_train-lag, size))
    q_test = np.zeros((n_test-lag, size))
    print("shape of q_train :", np.shape(q_train))
    for i in range(lag, n_train):
        q_train[i-lag][0] = P[i]
        q_train[i-lag][1] = P[i-1]
        q_train[i-lag][2] = P[i-2]
        q_train[i-lag][3] = P[i-3]
        q_train[i-lag][4] = P[i-4]
        q_train[i-lag][5] = E[i]
        
    for j in range(lag, n_test):
        q_test[j-lag][0] = P[j+n_train]
        q_test[j-lag][1] = P[j-1+n_train]
        q_test[j-lag][2] = P[j-2+n_train]
        q_test[j-lag][3] = P[j-3+n_train]
        q_test[j-lag][4] = P[j-4+n_train]
        q_test[j-lag][5] = E[j+n_train]

    return q_train, q_test


X_train, X_test = features(P, E, n_train, n_test)
Y_train, Y_test = h_train[4:][:], h_test[4:][:]

print("shape of X_train", np.shape(X_train))
print("shape of X_test", np.shape(X_test))
print("shape of Y_train", np.shape(Y_train))
print("shape of Y_test", np.shape(Y_test))


"""
# Random Forest
regr = RandomForestRegressor(n_estimators=50, criterion='mse', max_depth=None, min_samples_split=4, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, 
    verbose=0, warm_start=True)

regr.fit(X_train, Y_train)
h_hat = regr.predict(X_test)

print("shape of h_hat: ", np.shape(h_hat))
print("Feature importance :", regr.feature_importances_) 
print("Score :", regr.score(X_test, Y_test))

x = np.linspace(0, 3213, 3213)
#plot of
plt.figure(figsize=[12, 4])
plt.plot(x, h_hat[:,0])
plt.plot(x, Y_test[:,0])
plt.show()
"""



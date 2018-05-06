#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:13:18 2018

@author: thomas
"""

# -*- coding: utf-8 -*-
"""
Thomas Stolp

This is a temporary script file.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Laden van input gegevens:

observations = pd.read_pickle('../Data/meetraai_5.pickle') # Uurlijkse stijghoogtemetingen op locatie
forcing = pd.read_pickle('../Data/forcing_hourly_5.pickle') # Uurlijkse neerslag en verdamping



#plot of the piezometric levels in the dike 
plt.figure(figsize=[15, 4])
plt.plot(observations)
plt.show()

#plot of the percipation 
plt.figure(figsize=[15, 4])
plt.plot(forcing["neerslag"])
plt.xlabel("")
plt.ylabel("neerslag mm/uur")
plt.show()

#plot of the evaporation 
plt.figure(figsize=[15, 4])
plt.plot(forcing["verdamping"])
plt.xlabel("")
plt.ylabel("verdamping mm/uur")
plt.show()


#h = observations[34:11979][:] #without nan 
h = observations
P = forcing['neerslag']
E = forcing['verdamping']

print(np.shape(h))
print(np.shape(P))

# Response 
h_train = observations[37:8761][:]
h_test = observations[8761:12023][:]

train_samples = len(h_train)
test_samples = len(h_test)

'''

print(h_train[0][0])
for i in range(len(h_train)):
    for j in range(5):
        if h_train[i][j] == 'nan':
            print(i, j)
            observations[i][j] = observations[i-1][j]

def features(P, E):
    q = np.zeros((8761, 2))
    for i in range(8761):
        q[i][0] = P[i]
        q[i][1] = E[i]
    return q


X_train = features(P, E)
Y_train = h_train

regr = RandomForestRegressor(n_estimators=3, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, 
    verbose=0, warm_start=False)

regr.fit(X_train, Y_train)
print("Feature importance :", regr.feature_importances_) 

def featurestest(P, E):
    q = np.zeros((12023-8761, 2))
    for i in range(12023-8761):
        q[i][0] = P[i]
        q[i][1] = E[i]
    return q

X_test = featurestest(P, E)
h_hat = regr.predict(X_test)

'''

#plot of the piezometric levels in the dike 
plt.figure(figsize=[15, 4])
plt.plot(observations)
plt.show()

#plot of the percipation 
plt.figure(figsize=[15, 4])
plt.plot(forcing["neerslag"])
plt.xlabel("")
plt.ylabel("neerslag mm/uur")
plt.show()
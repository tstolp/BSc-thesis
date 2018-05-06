#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load data and remove Nan values

"""
import numpy as np

# data
observations = np.loadtxt('../Data/meetraai_5.txt') # Loading text files Uurlijkse stijghoogtemetingen op locatie
forcing = np.loadtxt('../Data/forcing_hourly_5.txt') # Loading text files Uurlijkse neerslag en verdamping

# First and last rows of observations contain nan values. 
n0 = 37 # first index with numbers 
n1 = 12014 # last index with numbers

P = forcing[n0:n1,0] # Neerslag in mm per uur
E = forcing[n0:n1,1] # Verdamping in mm per uur

h = observations[n0:n1, :] # stijghoogte op 5 plaatsen in het dijklichaam

N = 11977 

# Finding nan values ...
for i in range(N):
    for j in range(5):
        if np.isnan( h[i][j] ):
            h[i][j] = h[i-1][j]
            
#np.savetxt("h", h)
#np.savetxt("P", P)  
#np.savetxt("E", E)    

# plots of data

#plot of the piezometric levels in the dike 
plt.figure(figsize=[12, 4])
plt.plot(observations)
plt.show()

#plot of the percipation 
plt.figure(figsize=[12, 4])
plt.plot(forcing["neerslag"])
plt.show()

#plot of the evaporation 
plt.figure(figsize=[12, 4])
plt.plot(forcing["verdamping"])
plt.show()


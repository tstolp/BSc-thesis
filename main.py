# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# Laden van input gegevens:
observations = pd.read_pickle('../Data/meetraai_5.pickle') # Uurlijkse stijghoogtemetingen op locatie
forcing = pd.read_pickle('../Data/forcing_hourly_5.pickle') # Uurlijkse neerslag en verdamping

print(np.shape(observations))
print(np.shape(forcing))

print(observations)
#plt.plot(observations[0], observations[1])
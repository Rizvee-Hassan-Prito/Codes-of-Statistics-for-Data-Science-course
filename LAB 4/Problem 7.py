# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 03:32:56 2022

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

df1=pd.read_csv('dataset_lab04.csv')
plt.hist(df1['Class'], weights=np.ones(len(df1['Class'])) / len(df1['Class']))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:14:55 2022

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols


df=pd.read_csv("Datas.txt",sep="\t\t")
df_m=pd.melt(df.reset_index(),id_vars=['index'],value_vars=['FAF DU PLESSIS', 'MS DHONI','RAVI ASHWIN'])
df_m.columns=['index','Players','Runs'] 

print()
print(df)
print()
model = ols ('Runs ~ Players', data=df_m)
results=model.fit()
result_table=sm.stats.anova_lm(results,typ=2)
print(result_table)
print()

#%%
plt.subplot(2,1,1)
sns.boxplot(df_m['Players'],df_m['Runs'])
#%%
plt.subplot(2,1,1)
sns.swarmplot(df_m['Players'],df_m['Runs'])


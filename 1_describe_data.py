#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:20:12 2017

@author: yufan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)

np.random.seed(123)

# Turn off progress printing 
solvers.options['show_progress'] = False

#Assign Asset Classes
AC = np.array(["Date","stocks", "hedge funds", "government bonds", "real estate","money market","corporate bond","renew energy"])

#read return data
f = pd.read_excel("pythonindices.xlsx")

#Drop Days in YYYY-MM-DD
f['Datum'] = pd.to_datetime(f['Datum'], format = "%Y%m")
#f.index = f['Datum']
#f.drop(f.columns[0], axis = 1, inplace = True)
#f.index = f.index.map(lambda x: x.strftime('%Y-%m'))

#Drop last row of data - String
f = f[79:233]
#f = f[229:233]
#map asset class to index
f.columns = AC

print f.describe()

points_mus = f.mean()
points_sigma = f.std()
points_label = f.columns

def describeData():
    global f
    fig, axs = plt.subplots(ncols=7, figsize = (24,12))
    fig.tight_layout()
    sns.regplot(x=np.array(f.index),y="stocks",data = f,ax=axs[0])
    sns.regplot(x=np.array(f.index),y="hedge funds",data = f,marker='x',ax=axs[1])
    sns.regplot(x=np.array(f.index),y="government bonds",data = f,marker='+',ax=axs[2])
    sns.regplot(x=np.array(f.index),y="real estate", data = f, marker = '*', ax = axs[3])
    sns.regplot(x=np.array(f.index),y="money market", data = f, marker = 'o', ax = axs[4])
    sns.regplot(x=np.array(f.index),y="corporate bond", data = f, marker = 'x', ax = axs[5])
    sns.regplot(x=np.array(f.index),y="renew energy", data = f, marker = 'o', ax = axs[6])
    plt.show()
    
describeData()
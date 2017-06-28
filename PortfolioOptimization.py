# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:47:06 2017

@author: Yufan
"""
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
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
    st_plt = sns.regplot(x=np.array(f.index),y="stocks",data = f,ax=axs[0])
    hf_plt = sns.regplot(x=np.array(f.index),y="hedge funds",data = f,marker='x',ax=axs[1])
    gb_plt = sns.regplot(x=np.array(f.index),y="government bonds",data = f,marker='+',ax=axs[2])
    re_plt = sns.regplot(x=np.array(f.index),y="real estate", data = f, marker = '*', ax = axs[3])
    mm_plt = sns.regplot(x=np.array(f.index),y="money market", data = f, marker = 'o', ax = axs[4])
    cp_plt = sns.regplot(x=np.array(f.index),y="corporate bond", data = f, marker = 'x', ax = axs[5])
    re_plt = sns.regplot(x=np.array(f.index),y="renew energy", data = f, marker = 'o', ax = axs[6])
    plt.show()

## NUMBER OF ASSETS
n_assets = len(f.columns)

## NUMBER OF OBSERVATIONS
n_obs = len(f.index)

n_portfolios = 1000

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)
    
wei_list = []

for i in range(n_portfolios):
    wei_list.append(rand_weights(n_assets))

wei_list = np.sort(wei_list)

f.drop(f.columns[0], axis = 1, inplace = True)
returns = f.T

def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma


means, stds = np.column_stack([
    random_portfolio(returns) 
    for _ in xrange(n_portfolios)
])

def plotData():
    global stds, means, points_sigma, points_mus, f
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    plt.plot(stds, means, 'o', markersize=5)
    plt.plot(points_sigma, points_mus, 'o', color = 'r', markersize = 5)
    for i, text in enumerate(f.columns):
        ax.annotate(text,(points_sigma[i],points_mus[i]))
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.title('Mean and standard deviation of Asset Class returns From Datastream')
    plt.show()
  
def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 1000
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1)) 
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

weights, returns, risks = optimal_portfolio(returns)

def froPlot():
    global weights, returns, risks, stds, means, points_sigma, points_mus, f
    fig = plt.figure(figsize= (24,12))
    ax = fig.add_subplot(111)
    plt.plot(risks,returns,'y-o')
    plt.plot(stds, means, 'o')
    plt.plot(points_sigma, points_mus, 'o', color = 'r', markersize = 5)
    for i, text in enumerate(f.columns):
        ax.annotate(text,(points_sigma[i],points_mus[i]))

    plt.ylabel('mean')
    plt.xlabel('std')
    
    plt.show()
    
#==============================================================================
#     Generate the plots
#==============================================================================
describeData()
plotData()
froPlot()
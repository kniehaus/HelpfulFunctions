# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:21:38 2015

@author: trin2441
"""

# This is a script and set of functions for testing GPs
# simple implementation: just sq exp kernel
# 
# Written by KN, 1-Oct-2015


from __future__ import division
import math
import numpy as np
import matplotlib.pylab as plt
#import scipy
#from scipy.spatial.distance import pdist, squareform

##########################################################################
# Functions
##########################################################################


def kernel_sqExp(a,b, ls=1, sv=1):
    """
    Computes a squared exponential kernel for the input data
    
    sq exp = exp(-0.5*(ai-bj)^2) 
           = exp(-0.5*(a^2 + b^2 - 2ab))
   
    w/ hyperparams: 
    = sv * exp(-1/(2*ls^2) * (a-b)^2) + ns^2 * delta_fxn
     
    ----------
    
    Input
    
    a: array
        Numpy array of input values
        
    b: array
        Numpy array of input values
        
    ls: float (default=1)
        Length scale
        
    sv: float (default=1)
        Signal variance
        
    ----------
    
    Output
    
    my_kernel: array
        Squared exponential kernel evaluated on two input vectors
    
    Written by Kate Niehaus, 1-Oct-2015
    
    """
    a = a.T/ls
    b = b.T/ls
    D, n = np.shape(a)
    d, m = np.shape(b)
    sqdist = np.tile((a**2).T, [1, m]) + np.tile(b*b, [n, 1]) - 2*np.dot(a.T,b)
    my_kernel = (sv**2) * np.exp(-0.5*sqdist)
    
   #  written all out to illustrate (need to make sure a, b are in original dimensions):
#    my_kernel2 = np.zeros((n, m))
#    for i in range(n):
#        for j in range(m):
#            ai = a[i]
#            bj = b[j]
#            my_kernel2[i, j] = np.exp(-1/(2*ls**2) * (ai-bj)**2 )
#    my_kernel2 = my_kernel2 * (sv**2)
            
    return my_kernel

    
    


def returnNormal(cov, N1):
    """
    Returns a MV normal distribution, as generated from a generic
    normal generator, and with the input covariance
    
    ----------
    
    Input
    
    cov: array
        Kernel function (evaluated already)
        
    N1: int
        Number of evaluation functions to draw
        
    ----------
    
    Output
    
    new_prior: array
       MVN distribution with given covariance 
       
    
    Written by KN, 1-Oct-2015
    
    """
    offSet = 1e-3
    n1 = np.shape(cov)[0]
    decomp = np.linalg.cholesky(cov + offSet*np.eye(n1))
    new_prior = np.dot(decomp, np.random.normal(size=(n1,N1)))
    return new_prior
    
    
    
    
def computeRegression(Xin, Yin, Xtest, noise, ls, sv):
    """
    Computes GP regression for input x,y points, for given set of test 
    points and hyperparameters
    
    See pg. 19, Rasmussen & Williams, 2005
        
    ----------
    
    Input
    
    Xin: array
        x-values of data points you observe
    
    Yin: array
        y-values of data points you observe
    
    Xtest: array
        x-values of data points that you want to test on
    
    noise: float
        Noise variance; hyperparameter
    
    ls: float
        Length scale; hyperparameter
    
    sv: float
        Signal variance; hyperparameter
        
    ----------
    
    Output
    
    f_mean: array
        Mean of latent function, calculated at test points
    
    cov: array
        Covariance calculated for resulting GP
        
    lML: float
        Log-marginal likelihood for the model, given the set params
        
    lML_details: list
        Contains LML terms broken up into pieces: datafit, 
        complexity, and normalization term
    
    
    Written by KN, 1-Oct-2015
    """

    # compute kernels
    K = kernel_sqExp(Xin, Xin, ls=ls, sv=sv)
    Kstar = kernel_sqExp(Xin, Xtest, ls=ls, sv=sv)
    Kstarstar = kernel_sqExp(Xtest, Xtest, ls=ls, sv=sv)

    # compute mean based on training input points
    n1 = np.shape(K)[0]
    offSet = 1e-3
    L = np.linalg.cholesky(K + noise*np.eye(n1) + offSet*np.eye(n1))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L,Yin))
    f_mean = np.dot(Kstar.T,alpha)      # mean of points
    
    # compute resulting covariance of predictive distribution
    v = np.linalg.solve(L, Kstar)
    cov = Kstarstar - np.dot(v.T,v)
    
    # compute log of marginal likelihood
    #lML = -0.5*np.dot(Yin.T,alpha) - np.sum(np.log(L)) - (n1/2*np.log(2*math.pi))
    lML_dataFit = -0.5*np.dot(Yin.T,alpha)
    lML_complexity = -np.sum(np.log(L))
    lML_normalize = -(n1/2*np.log(2*math.pi))
    lML_details = [lML_dataFit, lML_complexity, lML_normalize]
    lML = lML_dataFit[0] + lML_complexity + lML_normalize 
    
    return f_mean, cov, lML , lML_details   




def computeCI(cov, mean):
    """
    Calculates the 95% confidence interval for a GP
    
    Based upon method on pg. 15 in Rasmussen, 2005    
    
    Input:
    - covariance: matrix of size nxn of covariances
    - mean: vector of size nx1 of means
    
    Output:
    - lower: vector of size nx1 of lower bounds
    - upper: vector of size nx1 of upper bounds
    
    
    ----------
    
    Input
    
    cov: array
        Covariance for test points; size nxn
        
    mean: array
        Mean of latent GP function at test points; size nx1
        
    ----------
    
    Output
    
    lower: array
        Lower bounds; size nx1
    
    upper: array
        Upper bounds; size nx1
       
    
    Written by Kate Niehaus, 20-May-2014
    
    """
    mult = 2
    sd = np.diag(cov)**(0.5)
    lower = mean - (mult*sd).reshape(-1,1)
    upper = mean + (mult*sd).reshape(-1,1)
    return lower, upper


#%%
##########################################################################
# Illustration of principle for prior distribution
##########################################################################

# Specify settings
np1 = 100            # number of evaluation points for functions 
xmin = 0            # range of x values to consider
xmax = 10

Np1 = 10             # number of example functions to draw from distribution

# Resulting vectors
Xtest = np.linspace(xmin, xmax, np1).reshape(-1,1)    # reshape makes into a n1 x 1 vector
cov_prior = kernel_sqExp(Xtest, Xtest)
draws_fromNormal = returnNormal(cov_prior, Np1)

# plot
#plt.figure()
#plt.subplot(1,2,1)
#plt.plot(Xtest, draws_fromNormal)


#%%
##########################################################################
# Illustration of principle if set the hyperparams yourself
##########################################################################


# Specify settings
ns1 = 100            # number of evaluation points for functions 
Ns1 = 10             # number of example functions to draw from distribution

ls = 1.5          # length scale
sv = 0.5          # signal variance

# set data points we observe [e.g. sine function]
ns2 = 30            # number of data points we see
Xin = np.linspace(xmin, xmax, ns2).reshape(-1,1)
Yin = np.sin(Xin) + np.random.random(np.shape(Xin))-0.5

# set hyperparams
noise = 0.3

# given hyperparams and data, compute distribution
f_mean, f_cov, lML, lMLdetails = computeRegression(Xin, Yin, Xtest, noise, ls, sv)

# get confidence intervals
lowerCI, upperCI = computeCI(f_cov, f_mean)

# make some draws from the posterior distribution of functions
draws_fromPosterior = np.dot(f_cov, np.random.normal(size=(ns1, Ns1))) + f_mean.reshape(ns1, -1)

# plot
# plot prior
plt.figure(figsize=[15,6])
plt.subplot(1,2,1)
plt.plot(Xin, Yin, 'ro', label='Input points', color='MidnightBlue')
plt.plot(Xtest, draws_fromNormal, linestyle='--')
plt.legend(loc='best', numpoints=1)
# plot posterior
plt.subplot(1,2,2)
#plt.figure(figsize=[8,5])
plt.plot(Xin, Yin, 'ro', label='Input points', color='MidnightBlue')
plt.plot(Xtest, f_mean, color='SteelBlue', linewidth=3, label='Learned function')
plt.plot(Xtest, np.sin(Xtest), color='Black', linewidth=3, label='True function')
plt.plot(Xtest, draws_fromPosterior, linestyle='--')
plt.plot(Xtest, upperCI, color='LightBlue', linewidth=3, label='95% CI')
plt.plot(Xtest, lowerCI, color='LightBlue', linewidth=3)
#plt.fill_between(Xtest, upperCI, lowerCI, where=None)
plt.legend(loc='best', numpoints=1)












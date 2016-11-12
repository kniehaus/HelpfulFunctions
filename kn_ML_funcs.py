# -*- coding: utf-8 -*-
"""
Created on Tue Feb 04 11:12:45 2014

Set of functions that deal with prediction 

NOTE: Needs cleaning!!!


@author: Kate Niehaus, 4-Feb-2014
"""

from __future__ import division

import matplotlib.pylab as plt
import matplotlib
import numpy as np

# mine
import kn_funcs as kn
import kn_specialized_funcs as kns

# sklearn
import sklearn as sk
from sklearn import svm
from sklearn.cluster import AgglomerativeClustering    
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.covariance import GraphLassoCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.naive_bayes import MultinomialNB


# scipy
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr

import math
import random






#######################################################################################
#
### Pre-processing functions
#
#######################################################################################    
    
    
    

def plotHistDists(featMat, trueLabels, featLabels):
    """
    Plots the distribution of the features for each of the groups
    
    NANs: if featMat contains NANs, they are ignored in plotting
    
    ----------
    
    Input
    
    featMat: numpy array
        Design matrix of data; (n examples) x (n features)
        
    trueLabels: list of 0s and 1s, length n examples
        List of labels (assumes binary labels of 0/1) for each example
        
    featLabels: list of strings, length n features
        List of strings to describe each feature
    
        
    ----------
    
    Output
    
    (plots each feature distribution)

    Written by KN, 27-Apr-2016
    
    """    
    plt.figure(figsize=[16,14])
    numR = np.ceil(len(featLabels)**0.5)
    for i, lab in enumerate(featLabels):
        plt.subplot(numR, numR, i+1)
        xRange = np.linspace(np.nanmin(featMat[:,i]), np.nanmax(featMat[:,i]), 25)
        g1 = featMat[trueLabels==0,i]
        g2 = featMat[trueLabels==1,i]
        plt.hist(g1[~np.isnan(g1)], bins=xRange, color='SteelBlue', alpha=0.5, normed=True)
        plt.hist(g2[~np.isnan(g2)], bins=xRange, color='FireBrick', alpha=0.5, normed=True)
        plt.xlabel(lab)
        plt.ylabel('Normalized count')
    plt.tight_layout()
    


def performChiSquared(featArr, trueLabels):
    """
    Calculates chi-squared tests for each feature
        
    ----------
    
    Input
    
    featArr: numpy array
        Design matrix of data; (n examples) x (n features)
        
    trueLabels: list of 0s and 1s, length n examples
        List of labels (assumes binary labels of 0/1) for each example
    
        
    ----------
    
    Output
    
    chi_squares: array
        Array of chi-squared statistics; length = n features
        
    p_values: array
        Array of associated p-values from chi-squared test; length = n features

    Written by Kate Niehaus, 12-Feb-2014
    
    """
    # get data into needed form
    susInd = np.where(trueLabels==0)[0]
    resistInd = np.where(trueLabels==1)[0]
    sumSus = np.sum(featArr[susInd, :], axis=0)     # observed class1
    sumResist = np.sum(featArr[resistInd,:], axis=0)    # observed class2
    meanCounts = (sumSus + sumResist)/2     # expected
    # ID and remove mutations that have too small expected counts
    cutoff = 5              # if expected freq is < 5, chi2 starts to become invalid
    ind_greaterThanCutoff = np.where(meanCounts>cutoff)[0]
    [pre_chi_squares, pre_p_values] = stats.chisquare([sumSus[ind_greaterThanCutoff], sumResist[ind_greaterThanCutoff]])
    # create filler arrays into which to insert the p-values that were calculated    
    chi_squares = np.zeros((len(meanCounts),1))
    p_values = np.ones((len(meanCounts),1))
    # fill-in calculated values
    i=0
    for ind in ind_greaterThanCutoff:
        chi_squares[ind] = pre_chi_squares[i]
        p_values[ind] = pre_p_values[i]
        i+=1    
    return chi_squares, p_values







def performPCA(featArr, numComp):
    """
    Performs PCA on an input zero-meaned matrix
    
    ----------
    
    Input
    
    featArr: numpy array
        Zero-meaned design matrix of data; (n examples) x (n features)
        
    numComp: int
        Number of components to return
    
        
    ----------
    
    Output
    
    reduc_featArr: array
        Input feature array reduced to the desired dimensionality 
        through PCA
        (num examples) x (numComp)
        
    exp_var: array
        Amount of variance in data explained by the number of components
        kept in 
        
    V: array
        Coefficients for each feature


    Written by Kate Niehaus, 28-Feb-2014
    
    """
    # with SVD
    U,s,V = np.linalg.svd(featArr)
    eigVals = s*s
    exp_var3 = np.sum(eigVals[0:numComp])/np.sum(eigVals)
    reduc_featArr3 = np.dot(featArr, V[0:numComp,:].T)
    return reduc_featArr3, exp_var3, V[0:numComp,:].T
    
     
    
    
def examineICA(featArr, my_n_components=2):
    """
    Performs ICA decomposition
    
    ----------
    
    Input
    
    featArr: numpy array
        Zero-meaned design matrix of data; (n examples) x (n features)
        
    my_n_components: int
        Number of components to return
    
        
    ----------
    
    Output
    
    sources: array
        Reduced feature array
    
    mixing: array
        Coefficients for each feature
    
    """
    my_max_iter=600
    my_ica = FastICA(max_iter = my_max_iter, n_components = my_n_components)
    sources = my_ica.fit_transform(featArr)
    mixing = my_ica.mixing_
    offset = my_ica.mean_
    return sources, mixing



def runProjection(df, projectionInput, visualizeVar, projOpt='PCA'):
    """
    Displays PCA/ICA projection onto 2-dimensions, colored according to input
    variable
    
    NAN treatment: replaces NANs with column mean
    
    ----------
    
    Input
    
    df: pandas data frame
        Contains data to be projected and coloring variable
        
    projectionInput: list of strings
        List of variables to include in projection
        
    visualizeVar: string
        Variable name to use for coloring
        
    projOpt: string (='PCA')
        Whether to use PCA ('PCA') or ICA ('ICA')
        
    ----------
    
    Output
    
    Plot with 2-d projection and input coloring choice
   
    Written by KN, 23-Sep-2016
    
    """
    data = df[projectionInput].values
    
    # replace nans with column mean
    colMean = np.nanmean(data, axis=1)
    nanInd = np.isnan(data)
    data[nanInd] = np.tile(colMean, ())
    
    # normalize 
    data = normalizeByCols(data)
    
    # get projection
    if projOpt=='PCA':
        reducFeatArr, expVar, coeff = performPCA(data,2)
    elif projOpt=='ICA':
        reducFeatArr, coeff = examineICA(data, my_n_components=2)
        
    # plot
    plt.figure(figsize=[8,6])
    #colorVar = [pt.timeFromBiol2EV(opt='relsurg', endTime = datetime.datetime(2015,11,30), crp_cutoff=crpCutoff)[0] if pt.timeFromBiol2EV(opt='relsurg', endTime = datetime.datetime(2015,11,30)) is not None else None for pt in ptObjsHere]
    #colorVar = np.array(colorVar)
    colorVar = df[visualizeVar].values
    plt.scatter(reducFeatArr[:,0], reducFeatArr[:,1], s=20, c=colorVar.reshape(len(colorVar),1), cmap='Blues')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()



def checkCov(inputMat, labels, title, plotOpt=1, returnOpt='spearman'):
    """
    Calculates the correlation between variables for the input 
    matrix
    
    NAN treatment: replaced with column mean
    
    Note: spearman corr is slightly different from np.corrcoef, which gives
    pearson correlation
    
    ----------
    
    Input
    
    inputMat: array
        Numpy array of data; design matrix
        
    labels: list
        List of names of features
        
    title: string
        Title for colormap graph of correlations
        
    plotOpt: int (default=1)
        Whether to display colormap graph (=1) or not (=0)
    
    returnOpt: string (default='spearman')
        Whether to use spearman (='spearman') or pearson (='pearson') 
        correlation in plot
    
        
    ----------
    
    Output
    
    corrMat: array
        Pearson correlations
        
    signTable: array
        p-values for spearman significance testing (uncorrected in any way) 
        
    corrMatS: array
        Spearman correlations

    
    Written by KN, 3-Mar-2016
    
    """
    numFeats = np.shape(inputMat)[1]
    # deal with NANs
    colMeans = np.nanmean(inputMat, axis=0)    
    meanMat = np.tile(colMeans, [np.shape(inputMat)[0],1])
    nanInd = np.isnan(inputMat)
    inputMat[nanInd] = meanMat[nanInd]
    # get correlation
    corrMat = np.corrcoef(inputMat.T)
    # get significance
    signTable = np.zeros((numFeats, numFeats))
    corrMatS = np.zeros((numFeats, numFeats))
    for i in range(numFeats):
        for j in range(numFeats):
            corr, p = spearmanr(inputMat[:,i], inputMat[:,j])
            signTable[i,j] = p
            corrMatS[i,j] = corr
    # plot
    if plotOpt==1:
        #plt.figure(figsize=[12,10])
        if returnOpt=='spearman':
            plt.pcolormesh(corrMatS); plt.colorbar()
        else:
            plt.pcolormesh(corrMat); plt.colorbar()
        plt.xticks(np.linspace(0.5, numFeats-0.5, numFeats), labels, rotation=270)
        plt.yticks(np.linspace(0.5, numFeats-0.5, numFeats), labels)
        plt.title(title)
        plt.tight_layout()

    return corrMat, signTable, corrMatS
    
    
    
    
def checkgraphlasso(inputMat, labels, title='', plotOpt=1):
    """
    Calculates the covariance as determiend through graphical 
    lasso for the input matrix, where any missing data is replaced with the 
    column mean
    
    NAN treatment: replaced with column mean
    
    Note: spearman corr is slightly different from np.corrcoef, which gives
    pearson correlation
    
    ----------
    
    Input
    
    inputMat: array
        Numpy array of data; design matrix
        
    labels: list
        List of names of features
        
    title: string
        Title for colormap graph of correlations
        
    plotOpt: int (default=1)
        Whether to display no graph (=0), colormap graph of graphical 
        lasso (=1), or comparison of graphical lasso, spearman, and 
        pearson correlations (=2)
    
        
    ----------
    
    Output
    
    cov_: array
        Graphical lasso covariance
        
    Prec_: array
        Graphical lasso precision
        
    CorrMat: array
        Correlations as assessed through pearson
        
    SignTable: 
        Table of significance as assessed through spearman r
    
    CorrMat2: 
        Correlations as assessed through spearman r

    
    Written by KN, 3-Mar-2016
    
    """
    numFeats = np.shape(inputMat)[1]
    
    # deal with NANs
    colMeans = np.nanmean(inputMat, axis=0)    
    colStd = np.nanstd(inputMat, axis=0)
    meanMat = np.tile(colMeans, [np.shape(inputMat)[0],1])
    stdMat = np.tile(colStd, [np.shape(inputMat)[0],1])
    nanInd = np.isnan(inputMat)
    inputMat[nanInd] = meanMat[nanInd]
    
    # standardize
    inputMat = inputMat - meanMat
    inputMat = inputMat/stdMat
    inputMat[np.isinf(inputMat)] = 0    # in case there is no variance in feature
    inputMat[np.isnan(inputMat)] = 0
    
    # get correlation
    corrMat = np.corrcoef(inputMat.T)
    # get significance
    signTable = np.zeros((numFeats, numFeats))
    corrMat2 = np.zeros((numFeats, numFeats))
    for i in range(numFeats):
        for j in range(numFeats):
            corr, p = spearmanr(inputMat[:,i], inputMat[:,j])
            signTable[i,j] = p
            corrMat2[i,j] = corr
    # get graphical lasso
    model = GraphLassoCV()
    model.fit(inputMat)
    cov_ = model.covariance_
    prec_ = model.precision_
    
    # plot
    if plotOpt==1:
        #plt.figure(figsize=[12,10])
        plt.pcolormesh(cov_); plt.colorbar()
        plt.xticks(np.linspace(0.5, numFeats-0.5, numFeats), labels, rotation=270)
        plt.yticks(np.linspace(0.5, numFeats-0.5, numFeats), labels)
        plt.title(title)
        plt.tight_layout()
    if plotOpt==2:
        matrices = [cov_, prec_, corrMat, corrMat2]
        matriceLabels = ['Graphical lasso cov', 'Graphical lasso prec', 'Pearson corr', 'Spearman corr']
        for i in range(len(matrices)):
            plt.subplot(2,2,i+1)
            plt.pcolormesh(matrices[i])
            plt.title(matriceLabels[i])
            plt.colorbar()
        plt.tight_layout()            
    return cov_, prec_, corrMat, signTable, corrMat2
    

#######################################################################################
#
### Performance evaluation functions
#
#######################################################################################


def calcPerf(trueLabels, predLabels, opt):
    """ 
    This function calculates the performance of a classifier, based upon the 
    comparison of the true and predicted labels
  
    ----------
    
    Input
    
    trueLabels: array
        numpy array of 1's and 0's of true labels
        
    predLabels: array
        numpy array of 1's and 0's of predicted labels
        
    opt: int
        What to return:
        opt=1. acc, sens, spec = calcPerf(trueLabels, predLabels, opt)
        opt=2. TP, TN, FP, FN = calcPerf(trueLabels, predLabels, opt)
        opt=3. false_pos_ind, false_neg_ind = calcPerf(trueLabels, predLabels, opt)
        
    ----------
    
    Output
    
    opt1: return acc, sens, spec proportions
    
    opt2: return TP, TN, FP, FN counts
    
    opt3: return the indices of FP and FN examples

    
    Written by Kate Niehaus, 6-Feb-2014
    
    """
    TP=FP=TN=FN=0
    false_pos_ind = []
    false_neg_ind = []
    if type(trueLabels)==np.float64:
        if predLabels==0 and trueLabels==0: TN = 1
        if predLabels==1 and trueLabels==0: FP = 1
        if predLabels==1 and trueLabels==1: TP = 1
        if predLabels==0 and trueLabels==1: FN = 1
    else:
        for i in xrange(len(trueLabels)):
            pred = predLabels[i]
            true = trueLabels[i]
            if pred==0 and true==0: TN = TN+1
            if pred==1 and true==0:
                FP = FP+1
                false_pos_ind.append(i)
            if pred==1 and true==1: TP = TP+1
            if pred==0 and true==1:
                FN = FN+1
                false_neg_ind.append(i)
        acc = float(TP + TN)/float(TP+FN+TN+FP)
        sens = float(TP)/float(TP+FN)
        spec = float(TN)/float(TN+FP)
    if opt==1:
        return(acc, sens, spec)
    elif opt==2:
        return(TP, TN, FP, FN)
    elif opt==3:
        return(false_pos_ind, false_neg_ind)
        
        

        


def computeSensSpecForROC(probs, testLabels, numPoints=40):
    """
    Varies the threshold cut-off and calculates the sensitivity
    and specificty that result
    
     ----------
    
    Input
    
    probs: array
        numpy array of probabilities of being in class
        
    testLabels: array
        numpy array of 1's and 0's of true labels
        
    numPoints: int
        Number of points in range [0,1] to use
        
    ----------
    
    Output
    
    acc: array
        Numpy array of accuracies as move through threshold range
        
    sens: array
        Numpy array of sensitivities as move through threshold range
        
    spec: array
        Numpy array of specificities as move through threshold range
    
    Written by KN, 9-Aug-2014
    
    """
    cutoffs = np.linspace(0,1,numPoints)
    acc = np.zeros((numPoints,1)).ravel()
    sens = np.zeros((numPoints,1)).ravel()
    spec = np.zeros((numPoints,1)).ravel()
    for i in range(numPoints):
        predictions = np.zeros((len(testLabels),1)).ravel()
        cutoff = cutoffs[i]
        yesPredInd = np.where(probs>cutoff)[0]
        noPredInd = np.where(probs<=cutoff)[0]
        predictions[yesPredInd] = 1
        acc[i],sens[i],spec[i] = calcPerf(testLabels, predictions, 1)
    return acc, sens, spec
    
    
    
def computeSensSpecForROC_LR(testSet, weights, testLabels, numPoints=40):
    """
    Varies the threshold cut-off and calculates the sensitivity
    and specificty that result, but written to accomodate the LR functions
    written previously
    
     ----------
    
    Input
    
    testSet: array
        numpy array of test data (n examples x n features)
        
    weight: array
        Numpy array of weights learned in LR classifier
        
    testLabels: array
        numpy array of 1's and 0's of true labels
        
    numPoints: int
        Number of points in range [0,1] to use
        
    ----------
    
    Output
    
    acc: array
        Numpy array of accuracies as move through threshold range
        
    sens: array
        Numpy array of sensitivities as move through threshold range
        
    spec: array
        Numpy array of specificities as move through threshold range
    
    Written by KN, 9-Aug-2014
    
    """
    cutoffs = np.linspace(0,1,numPoints)
    acc = np.zeros((numPoints,1)).ravel()
    sens = np.zeros((numPoints,1)).ravel()
    spec = np.zeros((numPoints,1)).ravel()
    #print(weights)
    for i in range(numPoints):
        predictions = np.zeros((len(testLabels),1)).ravel()
        cutoff = cutoffs[i]
        dotProd = np.dot(testSet, weights)
        probOfOne = [float(1)/(1+np.exp(-x)) for x in dotProd]
        probOfOne = np.array(probOfOne)
        resistInd = np.where(probOfOne>cutoff)[0]
        predictions[resistInd] = 1
        acc[i],sens[i],spec[i] = calcPerf(testLabels, predictions, 1)
    return acc, sens, spec



def projectRecursFeatRemoval_overallWeightings(weights, featLabelsIts, featLabels, includeString, N):
    """
    Puts weightings from each classifier after recursive feature removal 
    back into large matrix that contains all original features
    
    ----------
    
    Input

    weights: array (N subsamplings x num feats x num classifiers)
        Actual feature weightings for final classifier
        
    featlabelsIts: dictionary
        Features included as the final selected set for each 
        trial, for each algorithm; output of 
        performIterationsLoopRecursiveFeatElim()
        
    featLabels: list
        Vector/list of labels for each feature
    
    includeString: list
        Classifiers included in classification; defines order 
        of the weights
        
    N: int
        Number of iterations of subsampling
    
    ----------
    
    Output:
    
    overallFeatWeightings: array
        Array of zeros, with weightings for features filled 
        in when reach final selection
        (N subsamplings) x (len(features)) x len(includeString)
    
    Written by KN, 28-May-2015
    
    """
    # NOTE!!!! Hard-coded to reflect hard-coding order in 'performIterationsLoopRecursiveFeatElim'    
    # But all contained internally -- output is in order of includeString
    
    featLabelKeyString = ['RF', 'SVM', 'LR', 'LR_lasso']
    
    overallFeatWeightings = np.zeros((N, len(featLabels), len(includeString)))
    for i in range(N):
        for j in range(len(includeString)):
            key = featLabelKeyString.index(includeString[j])        # which index of featlabelsIts is needed to match with current classifier
            if includeString[j]!='SVM':
                for newFeat, newWeight in zip(featLabelsIts[key][i], weights[i,:,j]):
                    overallFeatWeightings[i,list(featLabels).index(newFeat),j] = newWeight
    return overallFeatWeightings
    
    
    
def makePtDictResults(probTable, idsList, includeString, classifierChoice):
    """
    Turns the probability table (output from performInterationsLoopGeneric) 
    into a condensed matrix of results for those patients who were ever in 
    the test set
    
    ----------
    
    Input
    - probTable: probability table to hold the probabilities of class 1 that 
    are predicted by each classifier 
        # for the test patients in each subsampling: 
        # (num IDs) x (num classifiers) x (N subsamplings)
    - classifierChoice: string indicating which classifier you want results for
    
    ----------
    
    Written by KN, 28-May-2015
    
    """
    idsList = np.array(idsList)
    # remove those patients who were never classified as test set
    mean1 = np.mean(probTable, axis=2)      # take mean by classifier
    mean2 = np.mean(mean1, axis=1)          # take mean by patient
    keepInd = np.where(mean2!=0)[0]
    probTable_reduced = probTable[keepInd,:,:]
    idList_reduced = idsList[keepInd]   
    # put into dictionary
    dictResults = {}
    for i, newid in enumerate(idList_reduced):
        resultsRow = probTable_reduced[i, includeString.index(classifierChoice),:]
        nonZeroInd = np.nonzero(resultsRow)
        dictResults[newid] = resultsRow[nonZeroInd]
    return dictResults, idList_reduced
    


def makeFeaturePlots(overallFeatWeightings, featLabels, includeString, classifierChoice, scatterOpt = 'off', newFig=True, rot=90):
    """   
    
    Plots the feature weightings for a given classifier as a set of violin 
    plots
    
    ----------
    
    Input

    - overallFeatWeightings: matrix of zeros, with weightings for features 
        filled in when reach final selection
        (N subsamplings) x (len(features)) x len(includeString)    
    - classifierChoice: string indicating which classifier you want results for   
    
    ----------
    
    Output
    
    """
    if newFig==True:
        plt.figure(figsize=[len(featLabels)*.5,10])
    xvals = np.linspace(0.5,len(featLabels)-.5, len(featLabels))
    
    data = overallFeatWeightings[:,:,includeString.index(classifierChoice)]
    
    if scatterOpt!='off':
        # plot as scatter plot
        xvals_project = np.tile(xvals, (np.shape(data)[0],1))
        plt.scatter(xvals_project, data)
    else:
        # plot those eligible as violin plot
        for i in range(len(featLabels)):
            subData = data[:,i]
            try:
                plotOutput = plt.violinplot(subData, [xvals[i]], showmedians=True)
                # change colors
                colorChoice = 'Indigo'
                for patch in plotOutput['bodies']:
                    patch.set_facecolor(colorChoice)
                for lineType in ['cmins', 'cmaxes', 'cbars', 'cmedians']:
                    plotOutput[lineType].set_edgecolor('Black')
            except Exception, ex:
                print(ex)
                xvals_project = xvals[i]*np.ones((1,len(subData)))
                plt.scatter(xvals_project, subData)
            
    # add feature labels
    plt.xticks(xvals, featLabels, rotation=rot)
    plt.hlines(0, xvals[0]-1, xvals[-1]+1, linestyle='--', color='Grey')
    plt.xlim(0,len(featLabels))
    if classifierChoice=='RF':
        plt.ylim(0,1)
    plt.grid()
    plt.ylabel('Feature weighting')
    plt.tight_layout()




def makeResultsViolinPlots(results, includeString):
    """ 
    Displays the distribution of AUROCs for the different classifiers
    
    Written by KN, 30-May-2015
    """
    xvals = np.linspace(0.5, len(includeString)-1, len(includeString))
    results_squashed = results[:,0,:]
    plotOutput = plt.violinplot(results_squashed, xvals, showmedians=True)
    for patch in plotOutput['bodies']:
        patch.set_facecolor('DarkGreen')
    for lineType in ['cmins', 'cmaxes', 'cbars', 'cmedians']:
        plotOutput[lineType].set_edgecolor('Black')
    plt.xticks(xvals, includeString)
    plt.grid()
    plt.ylabel('AUROC')
    plt.hlines(0.5, 0, len(includeString), 'r')
    plt.tight_layout()




def makePtViolinPlots(dictResults, dictLabels, UIlist):
    """
    Makes a set of violin plots for the patient prediction 
    results
    
    Goes from the UIlist so that patients are ordered by class
    
    Also, must use dictionary form because number of instances for each 
    patient varies b/c randomly in test/train set & subsampling
    
    ----------
    
    Input
    
    dictResults: dictionary
        Dictionary of [UIs --> list of results for classification
        probabilities], for given classifier (specified earlier)
        
    dictLabels: dictionary
        Dictionary of [UIs --> class label]
    
    UIlist: list
        List of UIs, with class0 first, then class1
    
    ----------
    
    Output
    
    Graph of violins, or points when there is only 1 data point, or no data
    when the patient was not ever included
    
    Written by KN, 29-May-2015
    
    """
    cutoff = 0.5    # cutoff for making predictions
    avrPrPt = np.zeros((len(UIlist),3))
    plt.figure(figsize=[25,5])
    counter=0.5
    for i, ui in enumerate(UIlist):
        # if this patient has any results
        if ui in dictResults.keys():
            # get class
            if dictLabels[ui]==0: colorChoice = 'Green'
            elif dictLabels[ui]==1: colorChoice= 'Blue'
            
            data = np.array(dictResults[ui])
            avrPrPt[i,0] = np.nanmean(data)
            avrPrPt[i,1] = dictLabels[ui]
            if np.nanmean(data)<cutoff:
                avrPrPt[i,2] = 0
            else:
                avrPrPt[i,2] = 1
            # if have just 1 data point
            if len(data)==1:
                plt.plot(counter,data[0], 'o', color=colorChoice)
            elif len(data)>1:
                # if have >1 data point
                try:
                    plotOutput = plt.violinplot(data, [counter], showmedians=True)
                    for patch in plotOutput['bodies']:
                        patch.set_facecolor(colorChoice)
                    for lineType in ['cmins', 'cmaxes', 'cbars', 'cmedians']:
                        plotOutput[lineType].set_edgecolor('Black')
                except:
                    plt.plot(counter,data[0], 'o', color=colorChoice)
        counter+=1
    plt.hlines(0.5, 0, counter, 'r')
    plt.ylabel('Classification score'); plt.xlabel('Patients')
    plt.tight_layout()
    
    return avrPrPt


#######################################################################################
#
### Getting test/training functions
#
#######################################################################################

def getTestTrain(featArr, trueLabels, comidsList, propTrain = 0.8):
    """
    Samples from larger class to produce training and test sets of equal size   
    
    ----------
    
    Input
    
    featArr: array
        Design matrix (n examples) x (n features)
        
    trueLabels: array
        Matrix of true labels (0s and 1s)
    
    comidsList: 
        Either list of IDs or [] empty array   
    
    ----------
    
    Output
    
    trainSet: array
        Balanced set of data for each class; 80% of size of smallest class
    
    testSet: array
        Balanced set of data for each class; 20% of size of smallest class
    
    trainLabels: Array
        Labels corresponding to training set
    
    testLabels: Array
        Labels corresponding to test set
    
    testComids: list
        List of comids corresponding to test set; if [] set passed
        in for comidsList input, then returns another [] empty set
    
    Written by Kate Niehaus, 7-Apr-2014
    
    """
    resistInd = np.where(trueLabels==1)[0]
    susInd = np.where(trueLabels==0)[0]
    
    # find which class is smaller and get subset from larger class
    if len(resistInd) <= len(susInd):    # if more susceptible samples,
        resistSet = featArr[resistInd,:]    # (use all resist data)
        susSubsetInd = random.sample(susInd,len(resistInd))                
        susSet = featArr[susSubsetInd,:]
        numSmaller = len(resistInd)
    else:       # if more resist samples, 
        susSet = featArr[susInd,:]    # (use all sus data)
        resistSubsetInd = random.sample(resistInd,len(susInd))                
        resistSet = featArr[resistSubsetInd,:]
        numSmaller = len(susInd)
        
    # split this into training and test sets
    #propTrain = 0.8
    nTrain = int(round(numSmaller*propTrain))
    nTest = int(len(resistSet)-nTrain)
    indices = np.random.permutation(numSmaller)
    trainInd, testInd = indices[:nTrain], indices[nTrain:]
    susTrain, susTest = susSet[trainInd,:], susSet[testInd,:]
    resistTrain, resistTest = resistSet[trainInd,:], resistSet[testInd,:]
    
    # concatenate together
    trainSet = np.concatenate((susTrain, resistTrain), axis=0)
    testSet = np.concatenate((susTest, resistTest), axis=0)
    trainLabels = np.concatenate((np.zeros((nTrain,1)), np.ones((nTrain,1))), axis=0)
    testLabels = np.concatenate((np.zeros((nTest,1)), np.ones((nTest,1))), axis=0) 
    trainLabels = np.ravel(trainLabels)
    testLabels = np.ravel(testLabels)
    
    # get comids
    testComids = []
    if len(comidsList) !=0:        
        comidsList = np.array(comidsList)
        try:
            testComids = kns.getComidsSelection(comidsList, susSubsetInd, testInd, resistInd)
        except UnboundLocalError:
            testComids = kns.getComidsSelection(comidsList, susInd, testInd, resistSubsetInd)
    return trainSet, testSet, trainLabels, testLabels, testComids
    


def getTestTrainFromObjects(ptObjs, propTest=0.2, setSeed=None):
    """
    Returns a random sample of the input list of objects as a training set
    and a testing set, with the numbers in each set by propTest
    
    Written by KN, 31-May-2015
    """
    if setSeed!=None:
        np.random.seed(setSeed)
    indices = np.random.permutation(len(ptObjs))
    print(indices)
    nTest = np.round(propTest*len(ptObjs))
    nTrain = len(ptObjs)-nTest
    trainInd, testInd = indices[:nTrain], indices[nTrain:]
    ptObjs = np.array(ptObjs)
    ptTrainSubset, ptTestSubset = ptObjs[trainInd], ptObjs[testInd]
    return list(ptTrainSubset), list(ptTestSubset)




def returnSubset(featArr, classInd, num):
    """
    Returns a random sample of length num from the section of the feature 
    array specified by the classInd
    
    Written by KN, 19-May-2015
    """
    class_subsetInd = random.sample(classInd, num)
    class_set = featArr[class_subsetInd,:]
    return class_set


def getTestTrain_3class(featArr, trueLabels):
    """
    trainSet, testSet, trainLabels, testLabels = getTestTrai_3class(featArr, trueLabels)
    
    Same as getTestTrain (returns balanced datasets), but for the 3-class 
    problem 
    
    Output
    - trainSet: balanced set of data for each class; 80% of size of smallest class
    - testSet: balanced set of data for each class; 20% of size of smallest class
    - trainLabels: labels corresponding to trainSet
    - testLabels: labels corresponding to test set
    
    Written by Kate Niehaus, 19-May-2015
    """
    trueLabels = np.array(trueLabels)
    class3Ind = np.where(trueLabels==2)[0]    
    class2Ind = np.where(trueLabels==1)[0]
    class1Ind = np.where(trueLabels==0)[0]
    # find which class is smaller and get subset from larger class
    classInds = [class1Ind, class2Ind, class3Ind]
    lengths = [len(class1Ind), len(class2Ind), len(class3Ind)]
    minlengthInd = np.argmin(lengths)    
    minSet = classInds[minlengthInd]
    # get subset from larger classes
    numSmaller = len(minSet)
    class1Set = returnSubset(featArr, class1Ind, numSmaller)
    class2Set = returnSubset(featArr, class2Ind, numSmaller)
    class3Set = returnSubset(featArr, class3Ind, numSmaller)
    # split this into training and test sets
    propTrain = 0.8
    nTrain = round(numSmaller*propTrain)
    nTest = numSmaller-nTrain
    indices = np.random.permutation(numSmaller)
    trainInd, testInd = indices[:nTrain], indices[nTrain:]
    train1, test1 = class1Set[trainInd,:], class1Set[testInd,:]
    train2, test2 = class2Set[trainInd,:], class3Set[testInd,:]
    train3, test3 = class3Set[trainInd,:], class3Set[testInd,:]
    # concatenate together
    trainSet = np.concatenate((train1, train2, train3), axis=0)
    testSet = np.concatenate((test1, test2, test3), axis=0)
    trainLabels = np.concatenate((np.zeros((nTrain,1)), np.ones((nTrain,1)), 2*np.ones((nTrain,1))), axis=0)
    testLabels = np.concatenate((np.zeros((nTest,1)), np.ones((nTest,1)), 2*np.ones((nTest,1))), axis=0) 
    trainLabels = np.ravel(trainLabels)
    testLabels = np.ravel(testLabels)
    return trainSet, testSet, trainLabels, testLabels
    
    
    
def getTestTrainNonEqual(featArr, trueLabels):
    """
    trainSet, testSet, trainLabels, testLabels, testComids = getTestTrain(featArr, trueLabels)
    
    Unlike getTestTrain, which returns balanced classes, getTestTrainNonEqual 
    returns a proportion of examples from each class without balancing
    
    Input
    - comidsList: optional.  Either list of comids or [] empty array    
    
    Output
    - trainSet: balanced set of data for each class; 80% of size of smallest class
    - testSet: balanced set of data for each class; 20% of size of smallest class
    - trainLabels: labels corresponding to trainSet
    - testLabels: labels corresponding to test set
    - testComids: list of comids corresponding to test set; if [] set passed
        in for comidsList input, then returns another [] empty set
    
    Written by Kate Niehaus, 30-oct-2014
    """
    resistInd = np.where(trueLabels==1)[0]
    susInd = np.where(trueLabels==0)[0]
    resistSet = featArr[resistInd,:]    # (use all resist data)
    susSet = featArr[susInd,:]    # (use all sus data)
    # split this into training and test sets
    propTrain = 0.8
    # sus group
    numSus = len(susInd)
    nTrainS = round(numSus*propTrain)
    nTestS = numSus-nTrainS
    indices = np.random.permutation(numSus)
    trainInd, testInd = indices[:nTrainS], indices[nTrainS:]
    susTrain, susTest = susSet[trainInd,:], susSet[testInd,:]
    # resist group
    numResist = len(resistInd)
    nTrainR = round(numResist*propTrain)
    nTestR = numResist-nTrainR
    indices = np.random.permutation(numResist)
    trainInd, testInd = indices[:nTrainR], indices[nTrainR:]
    resistTrain, resistTest = resistSet[trainInd,:], resistSet[testInd,:]
    # concatenate together
    trainSet = np.concatenate((susTrain, resistTrain), axis=0)
    testSet = np.concatenate((susTest, resistTest), axis=0)
    trainLabels = np.concatenate((np.zeros((nTrainS,1)), np.ones((nTrainR,1))), axis=0)
    testLabels = np.concatenate((np.zeros((nTestS,1)), np.ones((nTestR,1))), axis=0) 
    trainLabels = np.ravel(trainLabels)
    testLabels = np.ravel(testLabels)
    return trainSet, testSet, trainLabels, testLabels



#######################################################################################
#
### Univariate binary prediction (direct association)
#
#######################################################################################




def directAssocPredictExact(comidList, drug, knownMutDict, D_muts, D_mutPos):
    """ This function uses the direct association method on a given set of 
    mutation data to predict drug susceptibility
    
    Inputs:
    - comidList: list of comids to query
    - drug: drug of interest
    - knownMutDict: dictionary of generic-form known mutations, referencing 
        their non-generic form (if they have one) and their relevant drugs
        ie, pncA_*10* - pncA_Q10P, PZA
    - D_muts: dictionary of comids, referencing their non-generic mutations
        ie, comid - mutA1B, mutC2D
    - D_mutPos: dictionary of comids, referencing their generic mutations
        ie, comid - mut*1*, mut*2*
    
    Outputs:
    - predictions: numpy array of predictions
        0=susceptible, 1=resistant
    
    Written by Kate Niehaus, 4-Feb-2014
    """
    predictions = np.zeros((len(comidList),1))
    suspectedMuts = knownMutDict.keys()
    DA_mutList = []
    for b in suspectedMuts:
        if knownMutDict[b][1] ==drug:
            DA_mutList.append(b)
    # for each comid in list, 
    index=0
    for comid in comidList:
        for a in DA_mutList:
            if a in D_mutPos[comid]:
                # check if a is specific mutation
                if knownMutDict[a][0]==a:       # if non-specific
                    predictions[index] = 1
                else:
                    if knownMutDict[a][0] in D_muts[comid]:  # specific
                        print(knownMutDict[a][0])
                        predictions[index] = 1
        index+=1
    return predictions   
    

def runDA(testSet, testLabels, mutDAarr, opt):
    """
    results = runDA(testSet, testLabels, mutDAarr, opt)
    
    This function uses the Direct Association method to predict drug susceptibility
    
    Input:
    - testSet
    - testLabels
    - mutDAarr: vector of 1s and 0s indicating whether the corresponding 
        mutation feature is a suspected mutation or not (1=suspected, 0=not)
    - opt: option for assessing performance
        1=return acc, sens, spec; 2=return acc, TP, TN, FP, FN
    
    Output:
    - results: 
        opt1:3x1 vector containing accuracy, sensitivity, and specificity
        results for the given input data
        opt2:5x1 vector containing acc, TP, TN, FP, FN
        opt3: return just vector of predictions
    
    Written by Kate Niehaus, 5-Feb-2014 
    """
    mutDAarr = np.array(mutDAarr)
    # get indices of feature matrix that correspond to known & relevant drug mutations
    DA_ind = np.where(mutDAarr==1)[0]
    pred = np.zeros((testSet.shape[0],1))
    for i in xrange(testSet.shape[0]):
        if testSet[i,DA_ind].any()>0:
            pred[i] = 1
    if opt==3:
        return pred
    else:
        results = calcPerf(testLabels, pred, opt)
        return results



#######################################################################################
#
### Classifiers
#
#######################################################################################


def runLR_flex(el_ratio, loss='log', penalty='elasticnet'):
	"""
	This will run LR using an elastic net regularization (by default)
	- Can also change the loss ('hinge'=linear svm) and penalty ('l2' or 'l1')
	
	"""
	
	lrclassifier = SGDClassifier(loss=loss, penalty=penalty, alpha=0.0001, l1_ratio=el_ratio, fit_intercept=False, n_iter=5, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)



def predictLR(test, fitCoef, cutoff):
    """
    This function calculates the prediction of LR, given the coefficients and test data
    assign = predictLR(test, fitCoef, cutoff)
    
    Input:
    - test:
    - fitCoef:
    - cutOff:
    
    Output:
    - assign: binary predictions
    - probOfOne: probas
    
    Kate Niehaus, 24-Feb-2014
    """
    assign = np.zeros((len(test),1))
    #cutoff = 0.5        # if above this, call resistant.  otherwise, susceptible
    dotProd = np.dot(test, fitCoef)
    probOfOne = []
    for i in dotProd:
        probOfOne.append(1.0 / (1.0 + np.exp(-i)))
    probOfOne = np.array(probOfOne)
#    for i in probOfOne:
#        print('{0:.2f}'.format(probOfOne[i]))
    resistInd = np.where(probOfOne>cutoff)[0]
    assign[resistInd] = 1
    return assign, probOfOne


def runLR(trainSet,trainLabels, testSet, testLabels, Lnow, Tnow, penaltyOpt):
    """
    results, weights = runLRcv(trainSet,trainLabels, testSet, testLabels, skf)
    
    This function uses the LR algorithm with set parameters to fit 
    the data contained in the training set
    
    Input
    - trainSet, trainLabels, testSet, testLabels
    - Lnow: regularization parameter value
    - Tnow: cutoff parameter value 
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    - featWeight: numFeaturesx1 vector containing the weighting for each 
        mutation in the algorithm score
    
    Written by Kate Niehaus, 12-Mar-2014
    """
    # Predict, using set parameters, within test set
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    trainSet, testSet = normalizeByTraining(trainSet, testSet)    
    if penaltyOpt==1:       # Lasso regularisation
        pen = 'l1'
    elif penaltyOpt==2:     # ridge regression
        pen= 'l2'
    lrClassifier = LogisticRegression(C=Lnow, penalty=pen) 
    lrClassifier.fit(trainSet, np.ravel(trainLabels)) 
    weights = lrClassifier.coef_.ravel()
    pred, probs = predictLR(testSet, weights, Tnow)    
    auroc = roc_auc_score(testLabels, np.ravel(probs))
    results = calcPerf(testLabels, pred, 1)
    return auroc, weights, probs     


def runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penaltyOpt, numOutput=1):
    """
    results, weights = runLRcv(trainSet,trainLabels, testSet, testLabels, skf)
    
    This function runs cross-validation with the LR algorithm to fit 
    the data contained in the training set
    
    Input
    - trainSet, trainLabels, testSet, testLabels
    - skf: indices generated by StratifiedKFolds with which to run x-val
    - penaltyOpt: whether to use lasso (=1) or ridge (=2)
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    - featWeight: numFeaturesx1 vector containing the weighting for each 
        mutation in the algorithm score
    
    Written by Kate Niehaus, 20-Feb-2014
    """
    # set penalty option
    if penaltyOpt==1:       # Lasso regularisation
        pen = 'l1'
    elif penaltyOpt==2:     # ridge regression
        pen= 'l2'
    Crange = np.linspace(0.001,1,10)
    #cutOff = np.linspace(0.1, 1, 10)
    cutOff = [0.5]
    fold=0
    # initialize grid search results
    acc_grid = np.zeros((len(Crange),len(cutOff), 5))
    for ktrainInd, ktestInd in skf:
        # get folds
        ktrain = trainSet[ktrainInd]
        ktest = trainSet[ktestInd]
        ktrain, ktest = replaceNans_wColMean_trainTest(ktrain, ktest)
        ktrain, ktest = normalizeByTraining(ktrain, ktest)        
        ktrainLab = trainLabels[ktrainInd]
        ktestLab = trainLabels[ktestInd]
        # loop over grid of parameters:
        for j in xrange(len(Crange)):        
             for k in xrange(len(cutOff)):
                lrClassifier = LogisticRegression(C=Crange[j], penalty=pen)
                lrClassifier.fit(ktrain, np.ravel(ktrainLab))
                fitCoef = lrClassifier.coef_.ravel()
                kpred, kprobs = predictLR(ktest, fitCoef, cutOff[k])
                kauc = roc_auc_score(np.ravel(ktestLab), np.ravel(kprobs))
                kacc, ksens, kspec = calcPerf(np.ravel(ktestLab), np.ravel(kpred), 1)
                acc_grid[j,k, fold] = kauc
        fold+=1
    # compute maximum accuracy
    mean_acc_grid = np.mean(acc_grid, axis=2)
    max_acc = np.amax(mean_acc_grid)
    max_indices = np.where(mean_acc_grid==max_acc)       # note: 'where' returns row indices, then column indices
    # Use first instance of 'maximum' accuracy
    optC = Crange[max_indices[0][0]]
    optCutOff = cutOff[max_indices[1][0]]
    #print(optCutOff)
    # Predict, using optimal parameters, within test set
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    trainSet, testSet = normalizeByTraining(trainSet, testSet)    
    lrClassifier = LogisticRegression(C=optC, penalty=pen) 
    lrClassifier.fit(trainSet, np.ravel(trainLabels)) 
    weights = lrClassifier.coef_.ravel()
    pred, probs = predictLR(testSet, weights, optCutOff)       
    if numOutput==1:
        auroc = roc_auc_score(testLabels, np.ravel(probs))   
        return auroc, weights, probs        
    elif numOutput==4:
        results = calcPerf(testLabels, pred, 2)
        return results, weights, probs  
    else:
        results = calcPerf(testLabels, pred, 1)
        return results, weights, probs
        
        
    
def runLRcvRecursFeatElim(trainSet,trainLabels, testSet, testLabels, skf, penOpt, numEliminations, num2Elim, featLabels):
    """
    results, weights = runLRcvRecursFeatElim(trainSet,trainLabels, testSet, testLabels, skf, penOpt, numEliminations, num2Elim)
    
    This function runs cross-validation with the LR algorithm to fit 
    the data contained in the training set, with an added step of performing
    recursive feature elimination
    
    Input
    - trainSet, trainLabels, testSet, testLabels
    - skf: indices generated by StratifiedKFolds with which to run x-val
    - penaltyOpt: whether to use lasso (=1) or ridge (=2)
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    - featWeight: numFeaturesx1 vector containing the weighting for each 
        mutation in the algorithm score
    
    Written by Kate Niehaus, 28-Nov-2014
    """
    # set penalty option
    if penOpt==1:       # Lasso regularisation
        pen = 'l1'
    elif penOpt==2:     # ridge regression
        pen= 'l2'
    Crange = np.linspace(0.001,1,10)
    #cutOff = np.linspace(0.1, 1, 10)
    cutOff = [0.5]
    for i in range(numEliminations):
        fold=0
        # initialize grid search results
        acc_grid = np.zeros((len(Crange),len(cutOff), 5))
        for ktrainInd, ktestInd in skf:
            # get folds
            ktrain = trainSet[ktrainInd]
            ktest = trainSet[ktestInd]
            ktrain, ktest = replaceNans_wColMean_trainTest(ktrain, ktest)
            ktrain, ktest = normalizeByTraining(ktrain, ktest)
            ktrainLab = trainLabels[ktrainInd]
            ktestLab = trainLabels[ktestInd]
            # loop over grid of parameters:
            for j in xrange(len(Crange)):        
                 for k in xrange(len(cutOff)):
                    lrClassifier = LogisticRegression(C=Crange[j], penalty=pen)
                    lrClassifier.fit(ktrain, np.ravel(ktrainLab))
                    fitCoef = lrClassifier.coef_.ravel()
                    kpred, kprobs = predictLR(ktest, fitCoef, cutOff[k])
                    kauc = roc_auc_score(np.ravel(ktestLab), np.ravel(kprobs))
                    kacc, ksens, kspec = calcPerf(np.ravel(ktestLab), np.ravel(kpred), 1)
                    acc_grid[j,k, fold] = kauc
            fold+=1
        # compute maximum accuracy
        mean_acc_grid = np.mean(acc_grid, axis=2)
        max_acc = np.amax(mean_acc_grid)
        max_indices = np.where(mean_acc_grid==max_acc)       # note: 'where' returns row indices, then column indices
        # Use first instance of 'maximum' accuracy
        optC = Crange[max_indices[0][0]]
        optCutOff = cutOff[max_indices[1][0]]
        # normalize & remove nan's for temporary model fitting
        # (test set never used here)
        trainSet_temp, testSet_temp = replaceNans_wColMean_trainTest(trainSet, testSet)
        trainSet_temp, testSet_temp = normalizeByTraining(trainSet_temp, testSet_temp)        
        # Fit model
        lrClassifier = LogisticRegression(C=optC, penalty=pen) 
        lrClassifier.fit(trainSet_temp, np.ravel(trainLabels))
        # Get feature weightings
        weights = lrClassifier.coef_.ravel()  
        # Sort feature weightings
        sortInd = np.argsort(abs(weights))
        toKeep = sortInd[num2Elim:]
        # remove lowest abs(weightings) features from the dataset
        trainSet = trainSet[:,toKeep]
        testSet = testSet[:,toKeep] 
        featLabels = featLabels[toKeep]
    if numEliminations==0:
        fold=0
        # initialize grid search results
        acc_grid = np.zeros((len(Crange),len(cutOff), 5))
        for ktrainInd, ktestInd in skf:
            # get folds
            ktrain = trainSet[ktrainInd]
            ktest = trainSet[ktestInd]
            ktrain, ktest = replaceNans_wColMean_trainTest(ktrain, ktest)
            ktrain, ktest = normalizeByTraining(ktrain, ktest)            
            ktrainLab = trainLabels[ktrainInd]
            ktestLab = trainLabels[ktestInd]
            # loop over grid of parameters:
            for j in xrange(len(Crange)):        
                 for k in xrange(len(cutOff)):
                    lrClassifier = LogisticRegression(C=Crange[j], penalty=pen)
                    lrClassifier.fit(ktrain, np.ravel(ktrainLab))
                    fitCoef = lrClassifier.coef_.ravel()
                    kpred, kprobs = predictLR(ktest, fitCoef, cutOff[k])
                    kauc = roc_auc_score(np.ravel(ktestLab), np.ravel(kprobs))
                    kacc, ksens, kspec = calcPerf(np.ravel(ktestLab), np.ravel(kpred), 1)
                    acc_grid[j,k, fold] = kauc
            fold+=1
        # compute maximum accuracy
        mean_acc_grid = np.mean(acc_grid, axis=2)
        max_acc = np.amax(mean_acc_grid)
        max_indices = np.where(mean_acc_grid==max_acc)       # note: 'where' returns row indices, then column indices
        # Use first instance of 'maximum' accuracy
        optC = Crange[max_indices[0][0]]
        optCutOff = cutOff[max_indices[1][0]]
    #print(optCutOff)
    # Predict, using optimal parameters, within test set
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    trainSet, testSet = normalizeByTraining(trainSet, testSet)
    lrClassifier = LogisticRegression(C=optC, penalty=pen) 
    lrClassifier.fit(trainSet, np.ravel(trainLabels)) 
    weights = lrClassifier.coef_.ravel()
    pred, probs = predictLR(testSet, weights, optCutOff)    
    auroc = roc_auc_score(testLabels, np.ravel(probs))
    results = calcPerf(testLabels, pred, 1)
    return auroc, weights, featLabels, probs
    
    
 

def runMultiNB(trainSet, trainLabels, testSet, testLabels):
    """
    results = runSVMcv(trainSet, trainLabels, testSet, testLabels, skf)
    
    This function runs the svm algorithm with set parameters to fit 
    the data contained in the training set
    
    Input
    - trainSet, trainLabels, testSet, testLabels
    - Cnow: regularization parameter
    - Gnow: width of RBF kernel
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    
    Written by Kate Niehaus, 12-Mar-2014  
    """
    # Predict, using set parameters, within test set
    #trainSet, testSet = normalizeByTraining(trainSet, testSet)
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    multiNBClassifier = MultinomialNB(alpha=1, class_prior=None, fit_prior=False)
    multiNBClassifier.fit(trainSet, trainLabels)
    pred = multiNBClassifier.predict(testSet)
    results = calcPerf(testLabels, pred, 1)
    probas = multiNBClassifier.predict_proba(testSet)
    weights = multiNBClassifier.feature_log_prob_[0]
    auroc = roc_auc_score(testLabels, np.ravel(probas[:,1]))
    return auroc, weights, probas[:,1]   
   
    
    
def runSVM(trainSet, trainLabels, testSet, testLabels, Cnow, Gnow):
    """
    results = runSVMcv(trainSet, trainLabels, testSet, testLabels, skf)
    
    This function runs the svm algorithm with set parameters to fit 
    the data contained in the training set
    
    Input
    - trainSet, trainLabels, testSet, testLabels
    - Cnow: regularization parameter
    - Gnow: width of RBF kernel
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    
    Written by Kate Niehaus, 12-Mar-2014  
    """
    # Predict, using set parameters, within test set
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    trainSet, testSet = normalizeByTraining(trainSet, testSet)    
    svmClassifier = svm.SVC(C=Cnow, kernel='rbf', gamma=Gnow, probability=True, shrinking=True)
    svmClassifier.fit(trainSet, np.ravel(trainLabels))
    pred = svmClassifier.predict(testSet)
    results = calcPerf(testLabels, pred, 1)
    probas = svmClassifier.predict_proba(testSet)
    auroc = roc_auc_score(testLabels, np.ravel(probas[:,1]))
    return auroc, probas[:,1]   

    
def runSVMcv(trainSet, trainLabels, testSet, testLabels, skf, threshOpt, numOutput=1):
    """
    results = runSVMcv(trainSet, trainLabels, testSet, testLabels, skf)
    
    This function runs cross-validation with the svm algorithm to fit 
    the data contained in the training set
    
    Input
    - trainSet, trainLabels, testSet, testLabels
    - skf: indices generated by StratifiedKFolds with which to run x-val
    - threshOpt: whether to use probabilities as produced by classifier or not
        1=use probabilities and return these
        0 = don't and return acc, sens, spec as usual
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    
    Written by Kate Niehaus, 6-Feb-2014  
    """
    #Crange = np.logspace(0.0001,1,10)
#    Crange = np.linspace(0.001,10,10)
#    Grange = np.linspace(0.0001,10,10)
    numTestPoints = 20
    Crange = np.concatenate((np.linspace(0.001,1,int(numTestPoints/2)), np.logspace(0.001,1,int(numTestPoints/2))), axis=0)
    Grange = np.concatenate((np.linspace(0.001,1,int(numTestPoints/2)), np.logspace(0.001,1,int(numTestPoints/2))), axis=0)
    fold=0
    # initialize grid search results
    acc_grid = np.zeros((len(Crange),len(Grange), 5))
    for ktrainInd, ktestInd in skf:
        # get folds
        ktrain = trainSet[ktrainInd]
        ktest = trainSet[ktestInd]
        ktrain, ktest = replaceNans_wColMean_trainTest(ktrain, ktest)
        ktrain, ktest = normalizeByTraining(ktrain, ktest)
        #trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
        ktrainLab = trainLabels[ktrainInd]
        ktestLab = trainLabels[ktestInd]
        # loop over grid of parameters:
        for j in xrange(len(Crange)):        
             for k in xrange(len(Grange)):
                svmClassifier = svm.SVC(C=Crange[j], kernel='rbf', gamma=Grange[k], probability=True, shrinking=True)
                #svmClassifier = svm.SVC(C=Crange[j], kernel='linear', probability=False, shrinking=True)
                svmClassifier.fit(ktrain, np.ravel(ktrainLab))
                kpred = svmClassifier.predict(ktest)
                kprobas = svmClassifier.predict_proba(ktest)
                kauc = roc_auc_score(np.ravel(ktestLab), np.ravel(kprobas[:,1]))
                kacc, ksens, kspec = calcPerf(np.ravel(ktestLab), np.ravel(kpred), 1)
                acc_grid[j,k, fold] = kauc
        fold+=1
    # compute maximum accuracy
    mean_acc_grid = np.mean(acc_grid, axis=2)
    max_acc = np.amax(mean_acc_grid)
    max_indices = np.where(mean_acc_grid==max_acc)       # note: 'where' returns row indices, then column indices
    # Use first instance of 'maximum' accuracy
    optC = Crange[max_indices[0][0]]
    optG = Grange[max_indices[1][0]]
    # Predict, using optimal parameters, within test set
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    trainSet, testSet = normalizeByTraining(trainSet, testSet)    
    svmClassifier = svm.SVC(C=optC, kernel='rbf', gamma=optG, probability=True, shrinking=True, random_state=2)
    svmClassifier.fit(trainSet, np.ravel(trainLabels))
    pred = svmClassifier.predict(testSet)
    probas = svmClassifier.predict_proba(testSet)    
    if threshOpt==0 and numOutput==1:
        auroc = roc_auc_score(testLabels, np.ravel(probas[:,1]))
        return auroc, probas[:,1] 
    elif threshOpt==0 and numOutput==3:
        results = calcPerf(testLabels, pred, 1)
        return results, probas[:,1]
    elif threshOpt==0 and numOutput==4:
        results = calcPerf(testLabels, pred, 2)
        return results, probas[:,1]
    elif threshOpt==1:
        return probas[:,1]    
        
        
def runSVMcvRecursFeatElim(trainSet, trainLabels, testSet, testLabels, skf, threshOpt, numEliminations, num2Elim):     
    """
    results = runSVMcvRecursFeatElim(trainSet, trainLabels, testSet, testLabels, skf, threshOpt, numEliminations, num2Elim)
    
    This function runs cross-validation with the svm algorithm to fit 
    the data contained in the training set
    
    Input
    - trainSet, trainLabels, testSet, testLabels
    - skf: indices generated by StratifiedKFolds with which to run x-val
    - threshOpt: whether to use probabilities as produced by classifier or not
        1=use probabilities and return these
        0 = don't and return acc, sens, spec as usual
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    
    Written by Kate Niehaus, 6-Feb-2014  
    """
    # Crange = np.logspace(0.0001,1,10)
    #    Crange = np.linspace(0.001,10,10)
    #    Grange = np.linspace(0.0001,10,10)
    numTestPoints = 20
    Crange = np.concatenate((np.linspace(0.001,1,int(numTestPoints/2)), np.logspace(0.001,1,int(numTestPoints/2))), axis=0)
    Grange = np.concatenate((np.linspace(0.001,1,int(numTestPoints/2)), np.logspace(0.001,1,int(numTestPoints/2))), axis=0)
    for i in range(numEliminations):
        fold=0
        # initialize grid search results
        acc_grid = np.zeros((len(Crange),len(Grange), 5))
        for ktrainInd, ktestInd in skf:
            # get folds
            ktrain = trainSet[ktrainInd]
            ktest = trainSet[ktestInd]
            ktrain, ktest = replaceNans_wColMean_trainTest(ktrain, ktest)
            ktrain, ktest = normalizeByTraining(ktrain, ktest)            
            ktrainLab = trainLabels[ktrainInd]
            ktestLab = trainLabels[ktestInd]
            # loop over grid of parameters:
            for j in xrange(len(Crange)):        
                 for k in xrange(len(Grange)):
                    svmClassifier = svm.SVC(C=Crange[j], kernel='rbf', gamma=Grange[k], probability=False, shrinking=True)
                    #svmClassifier = svm.SVC(C=Crange[j], kernel='linear', probability=False, shrinking=True)
                    svmClassifier.fit(ktrain, np.ravel(ktrainLab))
                    kpred = svmClassifier.predict(ktest)
                    kprobas = svmClassifier.fit(ktrain, np.ravel(ktrainLab)).predict_proba(ktest)
                    kauc = roc_auc_score(np.ravel(ktestLab), np.ravel(kprobas[:,1]))
                    kacc, ksens, kspec = calcPerf(np.ravel(ktestLab), np.ravel(kpred), 1)
                    acc_grid[j,k, fold] = kauc
            fold+=1
        # compute maximum accuracy
        mean_acc_grid = np.mean(acc_grid, axis=2)
        max_acc = np.amax(mean_acc_grid)
        max_indices = np.where(mean_acc_grid==max_acc)       # note: 'where' returns row indices, then column indices
        # Use first instance of 'maximum' accuracy
        optC = Crange[max_indices[0][0]]
        optG = Grange[max_indices[1][0]]
        # normalize & remove nan's for temporary model fitting
        # (test set never used here)
        trainSet_temp, testSet_temp = replaceNans_wColMean_trainTest(trainSet, testSet)
        trainSet_temp, testSet_temp = normalizeByTraining(trainSet_temp, testSet_temp)        
        # Fit model
        svmClassifier = svm.SVC(C=optC, kernel='rbf', gamma=optG, probability=True, shrinking=True)
        svmClassifier.fit(trainSet_temp, np.ravel(trainLabels))
        # Get feature weightings
        featWeight = 0
        featWeight = 0
        # Sort feature weightings
        sortInd = np.argsort(abs(featWeight))
        toKeep = sortInd[num2Elim:]
        # remove lowest abs(weightings) features from the dataset
        trainSet = trainSet[:,toKeep]
        testSet = testSet[:,toKeep]    
    
    # Predict, using optimal parameters, within test set
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    trainSet, testSet = normalizeByTraining(trainSet, testSet)    
    svmClassifier = svm.SVC(C=optC, kernel='rbf', gamma=optG, probability=True, shrinking=True)
    svmClassifier.fit(trainSet, np.ravel(trainLabels))
    pred = svmClassifier.predict(testSet)
    results = calcPerf(testLabels, pred, 1)
    probas = svmClassifier.fit(trainSet, np.ravel(trainLabels)).predict_proba(testSet)
    auroc = roc_auc_score(testLabels, np.ravel(probas[:,1]))
    if threshOpt==0:
        return auroc, probas[:,1]        
    elif threshOpt==1:
        return probas     
    
    
    
        
        
        
def runRFcv(trainSet, trainLabels, testSet, testLabels, skf, threshOpt=0):
    """
    results, featWeight = runRFcv(trainSet, trainLabels, testSet, testLabels, skf)
    
    This function runs cross-validation with the RF algorithm to fit 
    the data contained in the training set
    
    Input
    - trainSet, trainLabels, testSet, testLabels
    - skf: indices generated by StratifiedKFolds with which to run x-val
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    - featWeight: numFeaturesx1 vector containing the weighting for each 
        mutation in the algorithm score
    
    Written by Kate Niehaus, 25-Feb-2014  
    """
    # parameters - grid search over
    my_n_estimators = np.linspace(20,60,5)
    my_prop_feat = np.linspace(0.3,0.7,5)
    my_criterion = 'entropy'
    fold=0
    # initialize grid search results
    acc_grid = np.zeros((len(my_n_estimators),len(my_prop_feat), 5))
    for ktrainInd, ktestInd in skf:
        # get folds
        ktrain = trainSet[ktrainInd]
        ktest = trainSet[ktestInd]
        ktrain, ktest = replaceNans_wColMean_trainTest(ktrain, ktest)
        ktrain, ktest = normalizeByTraining(ktrain, ktest)        
        ktrainLab = trainLabels[ktrainInd]
        ktestLab = trainLabels[ktestInd]
        # loop over grid of parameters:
        for j in xrange(len(my_n_estimators)):        
             for k in xrange(len(my_prop_feat)):
                 clf = RFC(n_estimators=int(my_n_estimators[j]), criterion=my_criterion, max_features=my_prop_feat[k]) 
                 clf.fit(ktrain, ktrainLab)
                 kpred = clf.predict(ktest)
                 kprobs = clf.predict_proba(ktest)
                 kauc = roc_auc_score(np.ravel(ktestLab), np.ravel(kprobs[:,1]))
                 kacc, ksens, kspec = calcPerf(np.ravel(ktestLab), np.ravel(kpred), 1)
                 acc_grid[j,k,fold] = kauc             
        fold+=1
    # compute maximum accuracy
    mean_acc_grid = np.mean(acc_grid, axis=2)
    max_acc = np.amax(mean_acc_grid)
    max_indices = np.where(mean_acc_grid==max_acc)       # note: 'where' returns row indices, then column indices
    # Use first instance of 'maximum' accuracy
    opt_n_estimators = my_n_estimators[max_indices[0][0]]
    opt_prop_feat = my_prop_feat[max_indices[1][0]]
    # Predict, using optimal parameters, within test set
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    trainSet, testSet = normalizeByTraining(trainSet, testSet)    
    clf = RFC(n_estimators=int(opt_n_estimators), criterion=my_criterion, max_features=opt_prop_feat) 
    clf.fit(trainSet, trainLabels)
    pred = clf.predict(testSet)
    results = calcPerf(testLabels, pred, 1)     # used as output previously
    featWeight = clf.feature_importances_
    featWeight = np.ravel(featWeight)
    probs = clf.predict_proba(testSet)
    auroc = roc_auc_score(testLabels, np.ravel(probs[:,1]))
    if threshOpt==0:
        return auroc, featWeight, probs[:,1]
    elif threshOpt==1:
        return probs 
        
    
def runRandomForest(trainSet, trainLabels, testSet, testLabels, my_n_estimators, my_prop_feat, numOutput=1):
    """
    results, featWeight = runRandomForest(trainSet, trainLabels, testSet, testLabels)
    
    This function runs the random forest algorithm to fit the data contained
    in the training set
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    - featWeight: numFeaturesx1 vector containing the weighting for each 
        mutation in the algorithm score
    
    Written by Kate Niehaus, 5-Feb-2014    
    """
    # options for RF
    # Normalize by training data
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    trainSet, testSet = normalizeByTraining(trainSet, testSet)    
    #my_n_estimators = 40        # number of trees in forest
    my_criterion = 'entropy'
    #my_prop_feat = 0.5      # number of variables/node: if a proportion, calculates prop*num features
    my_numFeat = np.round(my_prop_feat*np.shape(testSet)[1])
    clf = RFC(n_estimators=int(my_n_estimators), criterion=my_criterion, max_features=int(my_numFeat)) 
    clf.fit(trainSet, trainLabels)
    pred = clf.predict(testSet)    
    probs = clf.predict_proba(testSet)    
    featWeight = clf.feature_importances_
    featWeight = np.ravel(featWeight)
    if numOutput==1:
        auroc = roc_auc_score(testLabels, np.ravel(probs[:,1]))
        return auroc, featWeight, probs[:,1]
    elif numOutput==4:
        results = calcPerf(testLabels, pred, 2)         # used as output previously
        return results, featWeight, probs[:,1]
    else:
        results = calcPerf(testLabels, pred, 1)         # used as output previously
        return results, featWeight, probs[:,1]
    
    
def runRandomForestcvRecursFeatElim(trainSet, trainLabels, testSet, testLabels, skf, numEliminations, num2Elim, featLabels):
    """
    results, featWeight = runRandomForestcvRecursFeatElim(trainSet, trainLabels, testSet, testLabels, skf, numEliminations, num2Elim)
    
    This function runs cross-validation with the RF algorithm to fit 
    the data contained in the training set, with recursive feature elimination
    added in
    
    Input
    - trainSet, trainLabels, testSet, testLabels
    - skf: indices generated by StratifiedKFolds with which to run x-val
    - numEliminations: number of elimination rounds to perform
    - num2Elim: number of features to eliminate in each round
    
    Output:
    - results: 3x1 vector containing accuracy, sensitivity, and specificity
        results for the given round of training/testing
    - featWeight: numFeaturesx1 vector containing the weighting for each 
        mutation in the algorithm score
    
    Written by Kate Niehaus, 28-Nov-2014  
    """
    # parameters - grid search over
    my_n_estimators = np.linspace(20,60,5)
    my_prop_feat = np.linspace(0.3,0.7,5)
    my_criterion = 'entropy'
    for i in range(numEliminations):
        fold=0
        # initialize grid search results
        acc_grid = np.zeros((len(my_n_estimators),len(my_prop_feat), 5))
        for ktrainInd, ktestInd in skf:
            # get folds
            ktrain = trainSet[ktrainInd]
            ktest = trainSet[ktestInd]
            ktrain, ktest = replaceNans_wColMean_trainTest(ktrain, ktest)
            ktrain, ktest = normalizeByTraining(ktrain, ktest)            
            ktrainLab = trainLabels[ktrainInd]
            ktestLab = trainLabels[ktestInd]
            # loop over grid of parameters:
            for j in xrange(len(my_n_estimators)):        
                 for k in xrange(len(my_prop_feat)):
                     clf = RFC(n_estimators=int(my_n_estimators[j]), criterion=my_criterion, max_features=my_prop_feat[k]) 
                     clf.fit(ktrain, ktrainLab)
                     kpred = clf.predict(ktest)     
                     kprobs = clf.predict_proba(ktest)
                     kauc = roc_auc_score(np.ravel(ktestLab), np.ravel(kprobs[:,1]))
                     kacc, ksens, kspec = calcPerf(np.ravel(ktestLab), np.ravel(kpred), 1)
                     acc_grid[j,k,fold] = kauc                                   
            fold+=1
        # compute maximum accuracy
        mean_acc_grid = np.mean(acc_grid, axis=2)
        max_acc = np.amax(mean_acc_grid)
        max_indices = np.where(mean_acc_grid==max_acc)       # note: 'where' returns row indices, then column indices
        # Use first instance of 'maximum' accuracy
        opt_n_estimators = my_n_estimators[max_indices[0][0]]
        opt_prop_feat = my_prop_feat[max_indices[1][0]]
        # normalize & remove nan's for temporary model fitting
        # (test set never used here)
        trainSet_temp, testSet_temp = replaceNans_wColMean_trainTest(trainSet, testSet)
        trainSet_temp, testSet_temp = normalizeByTraining(trainSet_temp, testSet_temp)        
        # Fit model
        clf = RFC(n_estimators=int(opt_n_estimators), criterion=my_criterion, max_features=opt_prop_feat) 
        clf.fit(trainSet_temp, trainLabels)
        # Get feature weightings
        featWeight = clf.feature_importances_
        featWeight = np.ravel(featWeight)    
        # Sort feature weightings
        sortInd = np.argsort(abs(featWeight))
        toKeep = sortInd[num2Elim:]
        # remove lowest abs(weightings) features from the dataset
        trainSet = trainSet[:,toKeep]
        testSet = testSet[:,toKeep]
        featLabels = featLabels[toKeep]
    if numEliminations==0:
        fold=0
        # initialize grid search results
        acc_grid = np.zeros((len(my_n_estimators),len(my_prop_feat), 5))
        for ktrainInd, ktestInd in skf:
            # get folds
            ktrain = trainSet[ktrainInd]
            ktest = trainSet[ktestInd]
            ktrain, ktest = replaceNans_wColMean_trainTest(ktrain, ktest)
            ktrain, ktest = normalizeByTraining(ktrain, ktest)            
            ktrainLab = trainLabels[ktrainInd]
            ktestLab = trainLabels[ktestInd]
            # loop over grid of parameters:
            for j in xrange(len(my_n_estimators)):        
                 for k in xrange(len(my_prop_feat)):
                     clf = RFC(n_estimators=int(my_n_estimators[j]), criterion=my_criterion, max_features=my_prop_feat[k]) 
                     clf.fit(ktrain, ktrainLab)
                     kpred = clf.predict(ktest)
                     kprobs = clf.predict_proba(ktest)
                     kauc = roc_auc_score(np.ravel(ktestLab), np.ravel(kprobs[:,1]))
                     kacc, ksens, kspec = calcPerf(np.ravel(ktestLab), np.ravel(kpred), 1)
                     acc_grid[j,k,fold] = kauc           
            fold+=1
        # compute maximum accuracy
        mean_acc_grid = np.mean(acc_grid, axis=2)
        max_acc = np.amax(mean_acc_grid)
        max_indices = np.where(mean_acc_grid==max_acc)       # note: 'where' returns row indices, then column indices
        # Use first instance of 'maximum' accuracy
        opt_n_estimators = my_n_estimators[max_indices[0][0]]
        opt_prop_feat = my_prop_feat[max_indices[1][0]]
    # Predict, using optimal parameters, within test set
    trainSet, testSet = replaceNans_wColMean_trainTest(trainSet, testSet)
    trainSet, testSet = normalizeByTraining(trainSet, testSet)    
    clf = RFC(n_estimators=int(opt_n_estimators), criterion=my_criterion, max_features=opt_prop_feat) 
    clf.fit(trainSet, trainLabels)
    pred = clf.predict(testSet)
    results = calcPerf(testLabels, pred, 1)
    probs = clf.predict_proba(testSet)
    auroc = roc_auc_score(testLabels, np.ravel(probs[:,1]))
    featWeight = clf.feature_importances_
    featWeight = np.ravel(featWeight)
    return auroc, featWeight, featLabels, probs[:,1] 
    
    
    

#######################################################################################
#
### Normalizations
#
#######################################################################################
    
 
def normalizeByCols(designMatrix):
    """
    designMatrixNorm = normalizeByCols(designMatrix)

    Input:
    - designMatrix: raw design matrix

    Output: 
    - designMatrixNorm: normalized design matrix; normalized by the columns
    
    Written by Kate Niehaus, 23-June-2014
    """
    smallVal = 0.0001
    numEx, numFeat = np.shape(designMatrix)
    # get mean and std
    overallMean = np.nanmean(designMatrix, axis=0)
    overallStd = np.nanstd(designMatrix, axis=0)
    # find if there is anywhere the std = 0
    zeroInd = np.where(overallStd==0)
    overallStd[zeroInd] = smallVal*np.min(np.abs(designMatrix))
    # normalize matrix
    designMatrixNorm = designMatrix - np.tile(overallMean, (numEx,1))
    designMatrixNorm = designMatrixNorm/(np.tile(overallStd, (numEx,1)))
    return designMatrixNorm 
    
    
    
def normalizeByRows(designMatrix):
    """
    Normalizes input data by the row
    
    Written by KN, 21-Sep-2016
    
    """
    smallVal = 0.0001
    numEx, numFeats = np.shape(designMatrix)
    overallMeans = np.nanmean(designMatrix, axis=1).reshape(-1,1)
    overallStds = np.std(designMatrix, axis=1).reshape(-1,1)
    # find if there is anywhere the std = 0
    zeroInd = np.where(overallStds==0)
    overallStds[zeroInd] = smallVal*np.min(np.abs(designMatrix))
    # normalize matrix
    designMatrixNorm = designMatrix - np.tile(overallMeans, (1,numFeats))
    designMatrixNorm = designMatrixNorm/(np.tile(overallStds, (1,numFeats)))
    return designMatrixNorm 
     
    
    
def normalizeByTraining(trainSet, testSet): 
    """ 
    trainSet, testSet = normalizeByTraining(trainSet, testSet)

    Input:
    - trainSet: training data (n1xp)
    - testSet: testing data (n2xp)
    
    Output:
    - trainSetNorm: training data, normalized by training mean & std (n1xp)
    - testSetNorm: testing data, normalized by training mean & std (n2xp)
    
    Written by Kate Niehaus, 14-Feb-2014
    """
    smallVal = 0.0001
    numEx, numFeat = np.shape(trainSet)
    if len(np.shape(testSet))==1:
        numExTest = 1
    else:
        numExTest, numFeat = np.shape(testSet)
    means = np.mean(trainSet, axis=0)
    stds = np.std(trainSet, axis=0)   
    # set 0s to small value
    chInd = np.where(stds==0)[0]
    stds[chInd] = smallVal
    trainSetNorm = (trainSet - np.tile(means, (numEx,1)))/(np.tile(stds, (numEx,1)))
    testSetNorm = (testSet - np.tile(means, (numExTest,1)))/(np.tile(stds, (numExTest,1)))
    return trainSetNorm, testSetNorm
     
    
    
def replaceMissingWithMeanImputed(raw_data, ax=0):
    """
    This function will replace all NAN values with the mean of their 
    corresponding column (i.e., patient-wise mean)
    
    clean_mRNA_data = replaceMissingWithMeanImputed(all_mRNA_data, ax=0)
    
    Input: 
    - raw_data: 3d matrix of data with some NANs
    - ax: axis over which to compute means 
    
    Output
    - filled_data: 3d matrix of data with NANs replaced
    
    Written by KN, 19-Jan-2015
    """
    colMeans = np.nanmean(raw_data, axis=ax)
    meanMatrix = np.tile(colMeans, (np.shape(raw_data)[ax],1,1))
    nanInd = np.isnan(raw_data)
    clean_data = raw_data[:,:,:]
    clean_data[nanInd] = meanMatrix[nanInd]
    return clean_data
    
    
def replaceMissingWithMeanImputed2D(raw_data, ax=0):
    """
    This function will replace all NAN values with the mean of their 
    corresponding column (i.e., patient-wise mean)
    
    clean_mRNA_data = replaceMissingWithMeanImputed(all_mRNA_data, ax=0)
    
    Input: 
    - raw_data: 3d matrix of data with some NANs
    - ax: axis over which to compute means 
    
    Output
    - filled_data: 3d matrix of data with NANs replaced
    
    Written by KN, 2-Feb-2015
    """
    colMeans = np.nanmean(raw_data, axis=ax)
    meanMatrix = np.tile(colMeans, (np.shape(raw_data)[ax],1))
    nanInd = np.isnan(raw_data)
    clean_data = raw_data[:,:]
    clean_data[nanInd] = meanMatrix[nanInd]
    return clean_data        
    

def replaceNans_wColMean(featMat):
    """
    This function will replace any nans with the column mean
    If the entire column is Nans, will replace with zero
    """
    numFeat = np.shape(featMat)[1]
    cleanedFeatMat = featMat[:]
    for i in range(numFeat):
        column = featMat[:,i]
        nanInd = np.isnan(column)
        colMean = np.nanmean(column)
        if np.isnan(colMean):
            cleanedFeatMat[nanInd,i] = 0
        else:
            cleanedFeatMat[nanInd,i] = colMean
    return cleanedFeatMat
    
    
def replaceNans_wColMean_trainTest(trainMat, testMat):
    """
    cleanedFeatMat_train, cleanedFeatMat_test = replaceNans_wColMean_trainTest(trainMat, testMat)
    This function will replace any nans in both the test and training set 
    with the column mean from the training set
    
    If the entire column is Nans, will replace with zero
    """
    numFeat = np.shape(trainMat)[1]
    cleanedFeatMat_train = trainMat[:]
    cleanedFeatMat_test = testMat[:]
    for i in range(numFeat):
        column = trainMat[:,i]
        nanInd = np.isnan(column)
        colMean = np.nanmean(column)
        if np.isnan(colMean):
            colMean = 0
        # replace both
        nanInd_test = np.isnan(testMat[:,i])
        cleanedFeatMat_train[nanInd,i] = colMean
        cleanedFeatMat_test[nanInd_test,i] = colMean
    return cleanedFeatMat_train, cleanedFeatMat_test
    
    
    
#######################################################################################
#
### Functions to run many classifiers at once
#
#######################################################################################
    
    
    
#def runOptVals(trainSet,trainLabels, testSet, testLabels):
#    """    
#    results[i,:,n], weights[i,:,n] = runOptVals(trainSet,trainLabels, testSet, testLabels)     
#    
#    
#    
#    """
#    # Iterate through parameters
#    numTrials = 10
    


def varyNumInput(featArr, trueLabels, N, includeString):
    """
    resultsAddingData, numSamples = varyNumInput(featArr, trueLabels, N, includeString)
    
    This function varies the number of input samples and returns the prediction 
    performance as this is increased (to assess the bias vs. variance trade-off)
    
    Input:
    - featArr: array of feature values, as typical
    - trueLabels: array of 1's and 0's labeling each sample 
    - N: number of iterations to perform of subsamplings of larger group
    - includeString: techniques to use (e.g. ['SVM', 'LR'])
    
    Output:
    - resultsAddingData: array of results 
        (acc/sens/spec) x (num techniques) x (num samplings)
        e.g. (3) x (2 [SVM, LR]) x (10 [len(numSamples)])
    - numSamples: array of the x-values used as sample sizes
    
    Written by Kate Niehaus, 12-Mar-2014
    """
    resistInd = np.where(trueLabels==1)[0]
    susInd = np.where(trueLabels==0)[0]
    # get smaller sample size    
    if len(resistInd) < len(susInd): 
        smallerInd = list(resistInd)    # change to list so can shuffle
        largerInd = susInd
    else: 
        smallerInd= list(susInd)
        largerInd = resistInd
    # scramble smaller indices
    random.shuffle(smallerInd)
    smallerInd = np.array(smallerInd)       # change back to array
    # x-values - number of samples
    numSamples = map(int, np.linspace(10,len(smallerInd), 10))
    # initialize to hold results as num samples is varied
    varySize_results = np.zeros((3,len(includeString),len(numSamples))) 
    # select samples from dataset
    i=0
    for num in numSamples:
        featArr_mini = np.concatenate((featArr[largerInd,:], featArr[smallerInd[:num],:]))
        trueLabels_mini = np.concatenate((trueLabels[largerInd], trueLabels[smallerInd[:num]]))
        results, weights = performIterationsLoop(featArr_mini, trueLabels_mini, [], N, includeString)
        varySize_results[:,:,i] = (np.mean(results, axis=0))
        i+=1
    return varySize_results, numSamples


    

def performIterationsLoop(featArr, trueLabels, mutDAarr, N, includeString, comidsList):
    """
    results, weights = performIterationsLoop(featArr, trueLabels, mutDAarr, N, includeString, runCV)
    
    This function performs a machine learning round of learning and testing 
    for subsets of data sampled from the (larger) susceptible class
    
    Input:
    - featArr: numpy array of 0s up to 1s of size (num isolates) x (num mutations)
    - trueLabels: true labels for this dataset, of size (num isolates)
        0=susceptible, 1=resistant
    - mutDAarr: numpy array of 0s and 1s specifying whether the corresponding
        mutation is a "suspected" mutation of not; for Direct Association 
        comparison; of size (num mutations)
    - N: number of sub-samplings to perform
        
    Output:
    - results: (num iterations, numOutput, numTests) sized array containing
        accuracy, sensitivity, and specificity results for the N trials, with

    - weights: (num iterations, num mutations, numTests) sized array with a 
        weighting for each mutation for the N trials
        
        with: 
        -num iterations = N subsamples of susceptible dataset
        -numOutput = 3 (typically) for accuracy, sens, and spec results for 
            each N trial; could change by changing 'opt' param for assessPerf
        -numTests = the number of different algorithms tried 
            (e.g. 3 if comparing RF with SVM with DA)

    Written by Kate Niehaus, 4-Feb-2014    
    """   
    # split into resistant and susceptible subsets
    resistInd = np.where(trueLabels==1)[0]
    susInd = np.where(trueLabels==0)[0]
    DAind = np.where(mutDAarr==1)[0]
    # initialize
    numTests = len(includeString)        # e.g. DA, RF, etc
    numOutput = 3       # acc, sens, spec, typically
    results = np.zeros((N,numOutput,numTests))
    # above is a matrix of (num iterations) x (acc, TP, etc output) x (num types of classifier tried)
    # select for an iteration
    weights = np.zeros((N, np.shape(featArr)[1], numTests))
    # above is a matrix of (num iterations) x (weights) x (num classifiers with weights assigned)
    comidDict = {}
    trainSet, testSet, trainLabels, testLabels, testComids = getTestTrain(featArr, trueLabels, comidsList)
    
    for i in xrange(N):
        print(i)
        trainSet, testSet, trainLabels, testLabels, testComids = getTestTrain(featArr, trueLabels, comidsList)
        # split into x-validation folds
        k = 5
        skf = StratifiedKFold(trainLabels, k)
        #assigns[:,0] = knc.crossValidateAndPredict(kf, indAll, susTrain, susTest, resistTrain, resistTest)
        # perform machine learning optimization on these datasets  
        if 'DA' in includeString:
            # DA
            n = includeString.index('DA')
            results[i,:,n] = runDA(testSet, testLabels, mutDAarr, 1)
        if 'RF' in includeString:
            # RF
            n = includeString.index('RF')
            results[i,:,n], weights[i,:,n] = runRandomForest(trainSet, trainLabels, testSet, testLabels, 40, 0.5)
            #results[i,:,n], weights[i,:,n] = runRFcv(trainSet, trainLabels, testSet, testLabels, skf)
        if 'SVM' in includeString:
            # SVM
            n = includeString.index('SVM')
            results[i,:,n] = runSVMcv(trainSet, trainLabels, testSet, testLabels, skf)
        if 'LR' in includeString:
            # LR
            n = includeString.index('LR')
            penOpt = 2      # ridge regression penalty option
            results[i,:,n], weights[i,:,n] = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt)
        if 'LR_lasso' in includeString:
            n = includeString.index('LR_lasso')
            penOpt = 1      # lasso penalty option
            results[i,:,n], weights[i,:,n] = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt)
        if 'BNB' in includeString:
            # Bernoulli Naive Bayes
            n = includeString.index('BNB')
            results[i,:,n], BNBpred, probR, probS, theta_Rs, theta_Ss = BernoulliNaiveBayes(trainSet, trainLabels, testSet, testLabels, DAind)
            weights[i,:,n] = theta_Rs
#            # print out DA FN comids & BNB results
#            comidDict = kns.getFNprobabilities(np.ravel(DApred), testLabels, probR, testComids, comidDict)
#            # plot weights for BNB
#            xs = np.linspace(1,np.shape(testSet)[1], np.shape(testSet)[1])
#            plt.scatter(xs, theta_Rs[0], color='red')
#            plt.scatter(xs, theta_Ss[0], color='blue')
        if 'optVals' in includeString:
            # optimize values of features for cut-off prediction
            n = includeString.index('optVals')
            results[i,:,n], weights[i,:,n] = runOptVals(trainSet,trainLabels, testSet, testLabels)
    return results, weights





def performROCIterationsLoop(featArr, trueLabels, mutDAarr, N, includeString):
    """
    meanResults = performROCIterationsLoop(featArr, trueLabels, mutDAarr, N, includeString)
    
    This function performs a machine learning round of learning and testing 
    for subsets of data sampled from the larger class
    
    Input:
    - featArr: numpy array of 0s up to 1s of size (num isolates) x (num mutations)
    - trueLabels: true labels for this dataset, of size (num isolates)
        0=susceptible, 1=resistant
    - mutDAarr: numpy array of 0s and 1s specifying whether the corresponding
        mutation is a "suspected" mutation of not; for Direct Association 
        comparison; of size (num mutations)
    - N: number of sub-samplings to perform
        
    Output:
    - results: (numTestPoints, numTestPoints,numOutput, numTests) sized array containing
        accuracy, sensitivity, and specificity results averaged across N trials, 
        for each parameter value
        
        with: 
        -num iterations = N subsamples of susceptible dataset
        -numOutput = 3 (typically) for accuracy, sens, and spec results for 
            each N trial; could change by changing 'opt' param for assessPerf
        -numTests = the number of different algorithms tried 
            (e.g. 3 if comparing RF with SVM with DA)
        -numTestPoints = the number of different parameter values tried

    Written by Kate Niehaus, 4-Feb-2014    
    """   
    # split into resistant and susceptible subsets
    resistInd = np.where(trueLabels==1)[0]
    susInd = np.where(trueLabels==0)[0]
    # initialize
    numTests = len(includeString)        # e.g. DA, RF, etc
    numOutput = 3       # acc, sens, spec, typically
    numTestPoints = 20
    #SVM
    #Crange = np.linspace(0.001,10,int(numTestPoints)) 
    Crange = np.concatenate((np.linspace(0.001,1,int(numTestPoints/2)), np.logspace(0.001,1,int(numTestPoints/2))), axis=0)
    Grange = np.concatenate((np.linspace(0.0001,1,int(numTestPoints/2)), np.logspace(0.001,1,int(numTestPoints/2))), axis=0)
    #np.linspace(0.01,10,numTestPoints) # RBF width
    # LR     
    Lrange = np.linspace(0.001, 1, numTestPoints)  # degree of regularization
    Trange = np.linspace(0,1, numTestPoints)   # cutoff point
    #RF
    Nrange = np.linspace(20,60, numTestPoints)    # number of estimators
    Prange = np.linspace(0.25,0.75, numTestPoints)    # proportion of features to use
    # meanResults - matrix of (num param1) x (num param2) x (num types of classifier tried)
    meanResults = np.zeros((numTestPoints, numTestPoints,numOutput, numTests))
    for a in xrange(numTestPoints):
        print(a)
        for b in xrange(numTestPoints):
            # select for an iteration
            Cnow = Crange[a]
            Gnow = Grange[b]
            Lnow = Lrange[a]
            Tnow = Trange[b]
            Nest = Nrange[a]
            Pfeat = Prange[b]
            results = np.zeros((N,numOutput,numTests))
            for i in xrange(N):
                trainSet, testSet, trainLabels, testLabels, testComids = getTestTrain(featArr, trueLabels, [])
               # perform machine learning optimization on these datasets  
                if 'DA' in includeString:
                    # DA
                    n = includeString.index('DA')
                    results[i,:,n] = runDA(testSet, testLabels, mutDAarr, 1)
                if 'RF' in includeString:
                    # RF
                    n = includeString.index('RF')
                    results[i,:,n], weights = runRandomForest(trainSet, trainLabels, testSet, testLabels, Nest, Pfeat)
                if 'SVM' in includeString:
                    # SVM
                    n = includeString.index('SVM')
                    results[i,:,n] = runSVM(trainSet, trainLabels, testSet, testLabels, Cnow, Gnow)
                if 'LR' in includeString:
                    # LR
                    penOpt = 2
                    n = includeString.index('LR')
                    results[i,:,n], weights = runLR(trainSet,trainLabels, testSet, testLabels, Lnow, Tnow, penOpt)
                if 'LR_lasso' in includeString:
                    n = includeString.index('LR_lasso')
                    penOpt = 1      # lasso penalty option
                    results[i,:,n], weights = runLR(trainSet,trainLabels, testSet, testLabels, Lnow, Tnow, penOpt)
            # get mean of results across N iterations
            meanResults[a,b,:,:] = np.mean(results, axis=0)
    return meanResults
    

        




def recursiveFeatRemoval(featArr, trueLabels, mutList, mutDAarr, N):
    """
    resultsArr, finalWeights, featIncl = recursiveFeatRemoval(featArr, trueLabels, mutDAarr, N)
    
   Input:
    - featArr: numpy array of 0s up to 1s of size (num isolates) x (num mutations)
    - trueLabels: true labels for this dataset, of size (num isolates)
        0=susceptible, 1=resistant
    - 
    - mutDAarr: numpy array of 0s and 1s specifying whether the corresponding
        mutation is a "suspected" mutation of not; for Direct Association 
        comparison; of size (num mutations)
    - N: number of sub-samplings to perform
        
    Output:
    - resultsArr:
    - finalWeights: 
    - featIncl:

    Written by Kate Niehaus, 13-Feb-2014 
    """
    numRemovals = 20
    prop2Remove = 0.25
    avrResults = np.zeros((numRemovals, 3))
    mutList = np.array(mutList)
    for i in xrange(numRemovals):
        # run ML
        results, weights = performIterationsLoop(featArr, trueLabels, mutDAarr, N)
        # get averages & store acc, sens, spec
        avrResults[i,:] = np.mean(results[:,:,1], axis=0)
        avrWeights = np.mean(weights[:,:,1], axis=0)
        # sort & find lowest abs(weightings)
        sortInd = np.argsort(abs(avrWeights))
        toKeep = sortInd[round(prop2Remove*len(sortInd))+1:]
#        print(avrWeights[round(prop2Remove*len(sortInd))+1])
#        print(np.max(avrWeights))
        # remove lowest abs(weightings) features from the dataset
        featArr = featArr[:,toKeep]
        mutDAarr= mutDAarr[toKeep]
        mutList = mutList[toKeep]
        avrWeights = avrWeights[toKeep]
    # sort for easy viewing
    finalSortInd = np.argsort(avrWeights)
    mutList = mutList[finalSortInd]
    avrWeights = avrWeights[finalSortInd]
    return avrResults, avrWeights, mutList   
        
        
        
        
def getIDindices(testComids, comidsList):
    """
    testComidInd = getIDindices(testComids, comidsList)
    
    """
    testComids = list(testComids); comidsList = list(comidsList)
    indList = []
    for test in testComids:
        indList.append(comidsList.index(test))
    return np.array(indList)




def performIterationsLoopRecursiveFeatElim(featArr, trueLabels, N, includeString, featLabels, comidsList, numEliminations=5, num2Elim=15):
    """
    results, weights = performIterationsLoopRecursiveFeatElim(featArr, trueLabels, N, includeString)
    
    This function performs a machine learning round of learning and testing 
    for subsets of data sampled from the larger class.  It performs recursive 
    feature elimination within the training set for each subsampling. 
    
    Input:
    - featArr: numpy array of 0s up to 1s of size (num isolates) x (num mutations)
    - trueLabels: true labels for this dataset, of size (num isolates)
        0=susceptible, 1=resistant
    - N: number of sub-samplings to perform
    - includeString: e.g. ['SVM', 'LR']
    - numEliminations: how many iterations of feature removal to perform
    - num2Elim: number of features to eliminate in each step
        
    Output:
    - results: (num iterations, numOutput, numTests) sized array containing
        accuracy, sensitivity, and specificity results for the N trials, with

    - weights: (num iterations, num mutations, numTests) sized array with a 
        weighting for each mutation for the N trials
        
        with: 
        -num iterations = N subsamples of susceptible dataset
        -numOutput = 3 (typically) for accuracy, sens, and spec results for 
            each N trial; could change by changing 'opt' param for assessPerf
        -numTests = the number of different algorithms tried 
            (e.g. 3 if comparing RF with SVM with DA)

    Written by Kate Niehaus, 27-Nov-2014    
    """   
    # split into resistant and susceptible subsets
    resistInd = np.where(trueLabels==1)[0]
    susInd = np.where(trueLabels==0)[0]
    # initialize
    numTests = len(includeString)        # e.g. DA, RF, etc
    numOutput = 1       # acc, sens, spec, typically        OR 1 for auroc
    results = np.zeros((N,numOutput,numTests))
    # above is a matrix of (num iterations) x (acc, TP, etc output) x (num types of classifier tried)
    # select for an iteration
    weights = np.zeros((N, np.shape(featArr)[1]-numEliminations*num2Elim, numTests))
    # Also need to keep track of the names of the features selected
    featKeptRF = []
    featKeptSVM = []
    featKeptLR = []
    featKeptLRL1 = []    
    # above is a matrix of (num iterations) x (weights) x (num classifiers with weights assigned)
    probTable = np.zeros((len(comidsList), numTests, N))     
    # above is a probability table to hold the probabilities of class 1 that are predicted by each classifier 
        # for the test patients in each subsampling: 
        # (num IDs) x (num classifiers) x (N subsamplings) 
    
    for i in xrange(N):
        print(i)
        trainSet, testSet, trainLabels, testLabels, testComids = getTestTrain(featArr, trueLabels, comidsList)
        testComidInd = getIDindices(testComids, comidsList)  
        # split into x-validation folds
        k = 5
        skf = StratifiedKFold(trainLabels, k)
        #assigns[:,0] = knc.crossValidateAndPredict(kf, indAll, susTrain, susTest, resistTrain, resistTest)
        # perform machine learning optimization on these datasets  
        if 'RF' in includeString:
            # RF
            n = includeString.index('RF')
            results[i,:,n], weights[i,:,n], featKept_new, probTable[testComidInd,n,i] = runRandomForestcvRecursFeatElim(trainSet, trainLabels, testSet, testLabels, skf, numEliminations, num2Elim, featLabels)
            featKeptRF.append(featKept_new)            
            #results[i,:,n], weights[i,:,n] = runRFcv(trainSet, trainLabels, testSet, testLabels, skf)
        if 'SVM' in includeString:
            # SVM
            threshOpt = 0
            n = includeString.index('SVM')
            results[i,:,n], probTable[testComidInd,n,i] = runSVMcv(trainSet, trainLabels, testSet, testLabels, skf, threshOpt)
        if 'LR' in includeString:
            # LR
            penOpt = 2
            n = includeString.index('LR')
            results[i,:,n], weights[i,:,n], featKept_new, probTable[testComidInd,n,i]  = runLRcvRecursFeatElim(trainSet,trainLabels, testSet, testLabels, skf, penOpt, numEliminations, num2Elim, featLabels)
            featKeptLR.append(featKept_new)
        if 'LR_lasso' in includeString:
            n = includeString.index('LR_lasso')
            penOpt = 1      # lasso penalty option
            results[i,:,n], weights[i,:,n], featKept_new, probTable[testComidInd,n,i]  = runLRcvRecursFeatElim(trainSet,trainLabels, testSet, testLabels, skf, penOpt, numEliminations, num2Elim, featLabels)
            featKeptLRL1.append(featKept_new)
    featKept = [featKeptRF, featKeptSVM, featKeptLR, featKeptLRL1]
    return results, weights, featKept, probTable
    


def performIterationsLoopGeneric(featArr, trueLabels, N, includeString, comidsList):
    """
    results, weights = performIterationsLoop(featArr, trueLabels, N, includeString)
    
    This function performs a machine learning round of learning and testing 
    for subsets of data sampled from the (larger) susceptible class
    
    Input:
    - featArr: numpy array of 0s up to 1s of size (num isolates) x (num mutations)
    - trueLabels: true labels for this dataset, of size (num isolates)
        0=susceptible, 1=resistant
    - N: number of sub-samplings to perform
    - includeString: e.g. ['SVM', 'LR']
    - comidsList: list of example identifiers
        
    Output:
    - results: (num iterations, numOutput, numTests) sized array containing
        accuracy, sensitivity, and specificity results for the N trials, with

    - weights: (num iterations, num mutations, numTests) sized array with a 
        weighting for each mutation for the N trials
        
        with: 
        -num iterations = N subsamples of susceptible dataset
        -numOutput = 3 (typically) for accuracy, sens, and spec results for 
            each N trial; could change by changing 'opt' param for assessPerf
        -numTests = the number of different algorithms tried 
            (e.g. 3 if comparing RF with SVM with DA)

    Written by Kate Niehaus, 4-Feb-2014    
    """   
    # split into resistant and susceptible subsets
    resistInd = np.where(trueLabels==1)[0]
    susInd = np.where(trueLabels==0)[0]
    # initialize
    numTests = len(includeString)        # e.g. DA, RF, etc
    numOutput = 1       # acc, sens, spec, typically, OR 1 for AUROC
    results = np.zeros((N,numOutput,numTests))
    # above is a matrix of (num iterations) x (acc, TP, etc output) x (num types of classifier tried)
    # select for an iteration
    weights = np.zeros((N, np.shape(featArr)[1], numTests))
    # above is a matrix of (num iterations) x (weights) x (num classifiers with weights assigned)
    probTable = np.zeros((len(comidsList), numTests, N))     
    # above is a probability table to hold the probabilities of class 1 that are predicted by each classifier 
        # for the test patients in each subsampling: 
        # (num IDs) x (num classifiers) x (N subsamplings)
    
    for i in xrange(N):
        print(i)
        trainSet, testSet, trainLabels, testLabels, testComids = getTestTrain(featArr, trueLabels, comidsList)
        # normalize by training w/in x-validation loops
        # get indices of testComids
        testComidInd = getIDindices(testComids, comidsList)        
        
        # split into x-validation folds
        k = 5
        skf = StratifiedKFold(trainLabels, k)
        #assigns[:,0] = knc.crossValidateAndPredict(kf, indAll, susTrain, susTest, resistTrain, resistTest)
        # perform machine learning optimization on these datasets  
        if 'RF' in includeString:
            # RF
            n = includeString.index('RF')
            #results[i,:,n], weights[i,:,n], probTable[testComidInd,n,i] = runRandomForest(trainSet, trainLabels, testSet, testLabels, 40, 0.5)
            #results[i,:,n], weights[i,:,n], probTable[testComidInd,n,i] = runRandomForest(trainSet, trainLabels, testSet, testLabels, 40, 0.5)
            results[i,:,n], weights[i,:,n], probTable[testComidInd,n,i] = runRFcv(trainSet, trainLabels, testSet, testLabels, skf)
        if 'SVM' in includeString:
            # SVM
            threshOpt = 0
            n = includeString.index('SVM')
            results[i,:,n], probTable[testComidInd,n,i] = runSVMcv(trainSet, trainLabels, testSet, testLabels, skf, threshOpt)
        if 'LR' in includeString:
            # LR
            penOpt = 2
            n = includeString.index('LR')
            results[i,:,n], weights[i,:,n], probTable[testComidInd,n,i] = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt)
        if 'LR_lasso' in includeString:
            n = includeString.index('LR_lasso')
            penOpt = 1      # lasso penalty option
            results[i,:,n], weights[i,:,n], probTable[testComidInd,n,i] = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt)
        if 'Multi_NB' in includeString:
            n = includeString.index('Multi_NB')
            results[i,:,n], weights[i,:,n], probTable[testComidInd,n,i] = runMultiNB(trainSet, trainLabels, testSet, testLabels)

    return results, weights, probTable 

        

def performIterationsLoopGeneric_3Class(featArr, trueLabels, N, includeString, comidsList):
    """
    results, weights = performIterationsLoop(featArr, trueLabels, N, includeString)
    
    This function performs a machine learning round of learning and testing 
    for subsets of data sampled from the (larger) susceptible class
    
    Input:
    - featArr: numpy array of 0s up to 1s of size (num isolates) x (num mutations)
    - trueLabels: true labels for this dataset, of size (num examples)
        0=class1, 1=class2; 2=class3
    - N: number of sub-samplings to perform
    - includeString: e.g. ['SVM', 'LR']
    - comidsList: list of example identifiers
        
    Output:
    - results: (num iterations, numOutput, numTests) sized array containing
        accuracy, sensitivity, and specificity results for the N trials, with

    - weights: (num iterations, num mutations, numTests) sized array with a 
        weighting for each mutation for the N trials
        
        with: 
        -num iterations = N subsamples of susceptible dataset
        -numOutput = 3 (typically) for accuracy, sens, and spec results for 
            each N trial; could change by changing 'opt' param for assessPerf
        -numTests = the number of different algorithms tried 
            (e.g. 3 if comparing RF with SVM with DA)

    Written by Kate Niehaus, 4-Feb-2014    
    """   
    # initialize
    numTests = len(includeString)        # e.g. DA, RF, etc
    numOutput = 1       # acc, sens, spec, typically   OR 1 for auroc
    results = np.zeros((N,numOutput,numTests))
    # above is a matrix of (num iterations) x (acc, TP, etc output) x (num types of classifier tried)
    # select for an iteration
    weights = np.zeros((N, np.shape(featArr)[1], numTests))
    # above is a matrix of (num iterations) x (weights) x (num classifiers with weights assigned)
    
    for i in xrange(N):
        print(i)
        trainSet, testSet, trainLabels, testLabels = getTestTrain_3class(featArr, trueLabels)
        #trainSet, testSet = normalizeByTraining(trainSet, testSet)         (do later on within individual functions - b/c folds of x-val)
        # split into x-validation folds
        k = 5
        skf = StratifiedKFold(trainLabels, k)
        #assigns[:,0] = knc.crossValidateAndPredict(kf, indAll, susTrain, susTest, resistTrain, resistTest)
        # perform machine learning optimization on these datasets  
        if 'RF' in includeString:
            # RF
            n = includeString.index('RF')
            results[i,:,n], weights[i,:,n] = runRandomForest(trainSet, trainLabels, testSet, testLabels, 40, 0.5)
            #results[i,:,n], weights[i,:,n] = runRFcv(trainSet, trainLabels, testSet, testLabels, skf)
        if 'SVM' in includeString:
            # SVM
            threshOpt = 0
            n = includeString.index('SVM')
            results[i,:,n] = runSVMcv(trainSet, trainLabels, testSet, testLabels, skf, threshOpt)
        if 'LR' in includeString:
            # LR
            penOpt = 2
            n = includeString.index('LR')
            results[i,:,n], weights[i,:,n] = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt)
        if 'LR_lasso' in includeString:
            n = includeString.index('LR_lasso')
            penOpt = 1      # lasso penalty option
            results[i,:,n], weights[i,:,n] = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt)

    return results, weights    





def runLOO_3class(featArr, trueLabels, includeString, comidsList):
    """
    This is the outer loop for a supervised learning set-up with a single left-
    out patient
    
    No balancing is performed
    
    Written by KN, 26-Feb-2016
    """
    # initialize
    numPts = np.shape(featArr)[0]
    numTests = len(includeString)        # e.g. DA, RF, etc
    numOutput = 4       # acc, sens, spec, typically   OR 1 for auroc  OR TP, TN, FP, FN
    results = np.zeros((numPts,numOutput,numTests))
    # above is a matrix of (num iterations) x (acc, TP, etc output) x (num types of classifier tried)
    # select for an iteration
    weights = np.zeros((numPts, np.shape(featArr)[1], numTests))
    # above is a matrix of (num iterations) x (weights) x (num classifiers with weights assigned)    
    probTable = np.zeros((len(comidsList), numTests, numPts))     
    # above is a probability table to hold the probabilities of class 1 that are predicted by each classifier 
        # for the test patients in each subsampling: 
        # (num IDs) x (num classifiers) x (N subsamplings)
    
    trueLabels= np.array(trueLabels)
    trueLabels = np.ravel(trueLabels)
    
    for i in range(numPts):
        print(i)
        testComidInd= i        
        
        trainSet = np.concatenate((featArr[:i,], featArr[i+1:,]), axis=0)
        testSet = featArr[i,:]
        testSet = np.reshape(testSet, (len(testSet),1))
        trainLabels = np.concatenate((trueLabels[:i], trueLabels[i+1:]), axis=0)
        testLabels = trueLabels[i]
    
        # split into x-validation folds
        k = 5
        skf = StratifiedKFold(trainLabels, k)
        #assigns[:,0] = knc.crossValidateAndPredict(kf, indAll, susTrain, susTest, resistTrain, resistTest)
        # perform machine learning optimization on these datasets  
        if 'RF' in includeString:
            # RF
            n = includeString.index('RF')
            results[i,:,n], weights[i,:,n], probTable[testComidInd,n,i] = runRandomForest(trainSet, trainLabels, testSet, testLabels, 40, 0.5, numOutput)
            #results[i,:,n], weights[i,:,n] = runRFcv(trainSet, trainLabels, testSet, testLabels, skf)
        if 'SVM' in includeString:
            # SVM
            threshOpt = 0
            n = includeString.index('SVM')
            results[i,:,n], probTable[testComidInd,n,i] = runSVMcv(trainSet, trainLabels, testSet, testLabels, skf, threshOpt, numOutput)
        if 'LR' in includeString:
            # LR
            penOpt = 2
            n = includeString.index('LR')
            results[i,:,n], weights[i,:,n], probTable[testComidInd,n,i] = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt, numOutput)
        if 'LR_lasso' in includeString:
            n = includeString.index('LR_lasso')
            penOpt = 1      # lasso penalty option
            results[i,:,n], weights[i,:,n], probTable[testComidInd,n,i] = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt, numOutput)

    return results, weights 



    
    
def runSingleClassificationForROC(featArr, trueLabels, N, includeString, comidsList):
    """
    This averages across multiple runs to make an ROC curve
    
    Written by KN, 9-Aug-2014
    """
    resistInd = np.where(trueLabels==1)[0]
    susInd = np.where(trueLabels==0)[0]
    # initialize
    numTests = len(includeString)        # e.g. DA, RF, etc
    numOutput = 3       # acc, sens, spec, typically    OR 1 for auroc
    numPoints = 40
    results = np.zeros((N, numPoints, numOutput,numTests))
    # above is a matrix of (num iterations) x (num threshold points) x (acc, TP, etc output) x (num types of classifier tried)
    # select for an iteration
    weights = np.zeros((np.shape(featArr)[1], numTests))
    # above is a matrix of (num iterations) x (weights) x (num classifiers with weights assigned)
    comidDict = {}
    for i in xrange(N):
        print(i)
        trainSet, testSet, trainLabels, testLabels, testComids = getTestTrain(featArr, trueLabels, comidsList)
        # split into x-validation folds
        k = 5
        skf = StratifiedKFold(trainLabels, k)
        #assigns[:,0] = knc.crossValidateAndPredict(kf, indAll, susTrain, susTest, resistTrain, resistTest)
        # perform machine learning optimization on these datasets  
        #plt.figure()
        if 'SVM' in includeString:
            # SVM
            n = includeString.index('SVM')
            threshOpt = 1
            probs = runSVMcv(trainSet, trainLabels, testSet, testLabels, skf, threshOpt)
            results[i,:,0,n], results[i,:,1,n], results[i,:,2,n] = computeSensSpecForROC(probs[:,1], testLabels)    # acc, sens, spec
            #fpr, tpr, thresholds = roc_curve(testLabels, probs[:,1])
            #results[:,n] = auc(fpr, tpr)
            #plt.plot(1-spec, sens, color='red--', label='SVM')
        if 'LR' in includeString:
            # LR
            penOpt = 2
            n = includeString.index('LR')
            LRresults, weights, probs = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt)
            trainSetNorm, testSetNorm = normalizeByTraining(trainSet, testSet)
            results[i,:,0,n], results[i,:,1,n], results[i,:,2,n] = computeSensSpecForROC_LR(testSetNorm, weights, testLabels)
            #plt.plot(1-spec, sens, color='blue', label='LR ridge')
        if 'LR_lasso' in includeString:
            n = includeString.index('LR_lasso')
            penOpt = 1      # lasso penalty option
            LRresults, weights, probs = runLRcv(trainSet,trainLabels, testSet, testLabels, skf, penOpt)
            trainSetNorm, testSetNorm = normalizeByTraining(trainSet, testSet)
            results[i,:,0,n], results[i,:,1,n], results[i,:,2,n] = computeSensSpecForROC_LR(testSetNorm, weights, testLabels)
        if 'RF' in includeString:
            n = includeString.index('RF')
            threshOpt=1
            probs = runRFcv(trainSet, trainLabels, testSet, testLabels, skf, threshOpt)
            results[i,:,0,n], results[i,:,1,n], results[i,:,2,n] = computeSensSpecForROC(probs[:,1], testLabels)
            #plt.plot(1-spec, sens, color='green--', label='LR lasso')
    # calculate means
    meanResults = np.mean(results, axis=0)
    plt.figure(figsize=[10,10])
    if 'SVM' in includeString:
        # SVM
        n = includeString.index('SVM')
        if np.shape(featArr)[1]>10:
            plt.plot(1-meanResults[:,2,n], meanResults[:,1,n], color='Red', lw = 4, label='SVM')
        else:
            plt.plot(1-meanResults[:,2,n], meanResults[:,1,n], color='Pink', linestyle='--', lw = 4, label='SVM, only CRP')
    if 'LR' in includeString:
        n = includeString.index('LR')
        if np.shape(featArr)[1]>10:
            plt.plot(1-meanResults[:,2,n], meanResults[:,1,n], color='Blue', lw = 4, label='LR ridge')
        else:
            plt.plot(1-meanResults[:,2,n], meanResults[:,1,n], color='SteelBlue', linestyle='--', lw = 4, label='LR ridge, only CRP')
    if 'LR_lasso' in includeString:
        n = includeString.index('LR_lasso')
        if np.shape(featArr)[1]>10:
            plt.plot(1-meanResults[:,2,n], meanResults[:,1,n], color='Green', lw = 4, label='LR lasso')
        else:
            plt.plot(1-meanResults[:,2,n], meanResults[:,1,n], color='DarkGreen', linestyle='--', lw = 4, label='LR lasso, only CRP')
    if 'RF' in includeString:
        n = includeString.index('RF')
        if np.shape(featArr)[1]>10:
            plt.plot(1-meanResults[:,2,n], meanResults[:,1,n], color='Purple', lw = 4, label='RF')
        else:
            plt.plot(1-meanResults[:,2,n], meanResults[:,1,n], color='LightPurple', linestyle='--', lw = 4, label='RF, only CRP')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc="best")
    plt.grid()
    return results
    

    
#######################################################################################
#
### BMM
#
#######################################################################################    
   
    
    
def calcCompleteDataLL(mu, posterior, pi, featArr):
    """llikelihood = calcCompleteDataLL(opt_mus, posterior, pi_k[it+1], featArr)    
    
    
    """
    K, D = np.shape(mu)        # number of groups, number of features
    N = np.shape(featArr)[0]        # number of examples
    intSum = 0
    for n in range(N):      # for all examples n,
        for k in range(K):      # for all groups k, 
            Dsum = 0
            for i in range(D):      # for all features i:
                Dsum = Dsum + featArr[n,i]*np.log(mu[k,i]) + (1-featArr[n,i])*np.log(1-mu[k,i])
            intSum = intSum + posterior[k,n]*(np.log(pi[k]) + Dsum)
    return intSum
    
    

    
def EM_BernoulliMM(featArr, k, input_mus):
    """
    opt_mus, evidence, posterior, llikelihood = EM_BernoulliMM(featArr, k)

    Input
    - featArr: feature array of 1s and 0s
        (num examples x num features)
    - k: number of mixes assumed

    Output:
    - opt_mus: optimized mu values for each feature for each mix
        (k x num features)
    - evidence: summed total evidence across all samples
    - posterior: posterior probability of each class for each sample
        (k x num examples)
    
    Written by Kate Niehaus, 26-Mar-2014; edited 9-Apr-2014
    """
    num_its = 30
    epsilon = 0.01
    numEx = np.shape(featArr)[0]
    numFeat = np.shape(featArr)[1]
    # initialize
    if input_mus != []:
        opt_mus = input_mus
    else:
        opt_mus = np.random.random_sample((k, numFeat))    # random initialization on interval [0,1)  
    pi_k = np.zeros((k, num_its))                   #  k x num iterations
    pi_k[:,0] = float(1)/k         # prior is equal probability of being in any group
    llikelihood = 0     # loglikelihood

    evidence_old = np.zeros((1,numEx))
    for it in range(num_its-1):
        # E-step
        # calculate posterior prob of component k for each data point
        numerator = np.zeros((k,numEx))     #  k x num examples
        for i in range(k): 
            for j in range(numEx):
                numerator[i,j] = pi_k[i,it]*kn.computeBinomProb(opt_mus[i,:], featArr[j,:])
        evidence_new = np.sum(numerator,axis=0)     # should be of size (1 x num examples)
        posterior = numerator/np.tile(evidence_new, (k,1))    # should be of size (k x num examples)
        # handy to calculate:        
        Nks = np.sum(posterior,axis=1)      # should be of size (k x 1) # number of samples in each class
        xks = np.zeros((k,numFeat))           # size (k x num features)
        for i in range(k):  
            newSum = np.zeros((1,numFeat))      # size (1 x num features)
            for j in range(numEx):
                newSum += (posterior[i,j]*featArr[j,:])
            xks[i,:] = newSum/Nks[i]
        # Check for convergence
        if it>2:
            if np.abs(np.sum(xks[:]-opt_mus))<epsilon:
                break
            if np.sum(evidence_new-evidence_old)<epsilon:
                break
            if math.isnan(llikelihood):
                break
        # M-step
        opt_mus = xks[:]        # when maximize LL, get that the optimal mu values = xk's calculated earlier
        pi_k[:,it+1] = Nks/numEx
        evidence_old = evidence_new[:]
        # calculate complete-data log likelihood
        llikelihood = calcCompleteDataLL(opt_mus, posterior, pi_k[:,it+1], featArr)
        print(llikelihood)
    return opt_mus, np.sum(evidence_new), posterior, llikelihood, pi_k[:,it]
    
    
def EM_Bernoulli(trainSet, trainLabels, testSet, testLabels, k):
    """
    results, probR, probS, opt_musR, opt_musS = EM_Bernoulli(trainSet, trainLabels, testSet, testLabels, k)
    
    This function fits a Bernoulli mixture model to the input training data
    with k mixtures for each class, and then it predicts using the model
    on the test data.     
    
    Input:
    - trainSet
    - trainlabels
    - testSet
    - testLabels
    - k: number of mixes for Bernoulli mixture model
    
    Output:
    - results: acc, sens, spec
    - probR: probability of class y=1 for each data point
    - probS: probability of class y=0 for each data point
    - opt_musR: optimized mu values for the y=1 class
    - opt_musS: optimized mu values for the y=0 class
    
    Written by Kate Niehaus, 27-Mar-2014
    """
    classR = trainSet[:,np.where(trainLabels==1)[0]]
    classS = trainSet[:,np.where(trainLabels==0)[0]]
    # fit for each class:
    opt_musR, evidenceR, posteriorR = EM_BernoulliMM(classR, k)
    opt_musS, evidenceS, posteriorS = EM_BernoulliMM(classS, k)
    #print(opt_musR)
    #print(opt_musS)
    # test on test set
    probR = np.zeros((len(testLabels),1))
    probS = np.zeros((len(testLabels),1))
    pred = np.zeros((len(testLabels),1))
    for i in range(len(testLabels)):
        sumR = 0
        sumS = 0
        for j in range(k):
            sumR += kn.computeBinomProb(opt_musR[j,:],testSet[i,:])
            sumS += kn.computeBinomProb(opt_musS[j,:], testSet[i,:])
        probR[i] = sumR
        probS[i] = sumS
        if probR[i]>probS[i]: pred[i] = 1
    results = calcPerf(testLabels, pred, 1)
    return results, probR, probS, opt_musR, opt_musS
    
    
    
def fitBernoulliNaiveBayes(trainSet, trainLabels, DAind, distOpt):
    """
    theta_Rs, thetaSs, prior_R, prior_S = fitBernoulliNaiveBayes(trainSet, trainLabels, DAind)
    
    Input:
    - trainSet: array of (num ex) x (num features) of binary data
    - trainLabels: array of 1's and 0's indicating the class labels
        (num ex x 1)
    - DAind: list of indices of 'suspected' mutations
    - distOpt: whether to return parameters for distribution over thetas
        1=yes, return them; 0=no, do not return them
    
    Output:
    - theta_Rs: fitted theta parameters for R class
    - theta_Ss: fitted theta parameters for S class
    - prior_R: prior for R class
    - prior_S: prior for S class
    - [newAs_R, newAs_S, newBs_R, newBs_S]: parameters for posterior beta 
        distributions over each theta for each class
    
    Written by Kate Niehaus, 7-Apr-2014    
    """
    numFeatures = np.shape(trainSet)[1]
    # set priors
    # beta
    aS = np.ones((1,numFeatures))
    aR = aS[:]
    aR[0,DAind] = 1
    aS[0,DAind] = 0.25
    bS = np.ones((1,numFeatures))
    bR = bS[:]
    bR[0,DAind] = 0.25
    bS[0,DAind] = 1
    # dirichlet
    alphaR = np.sum(trainLabels)/len(trainLabels) 
    alphaS = 1-alphaR
    # split into classes
    classR = trainSet[np.where(trainLabels==1)[0],:]
    classS = trainSet[np.where(trainLabels==0)[0],:]
    # get Njs
    Nj_Rs = np.sum(classR, axis=0)*np.ones((1,(np.shape(trainSet)[1])))     # number of examples with each feature
    Nj_Ss = np.sum(classS, axis=0)*np.ones((1,(np.shape(trainSet)[1])))
    Nc_R = np.shape(classR)[0]      # number of resistant
    Nc_Rs = Nc_R*np.ones((1,(np.shape(trainSet)[1])))
    Nc_S = np.shape(classS)[0]      # number of susceptible
    Nc_Ss = Nc_S*np.ones((1,(np.shape(trainSet)[1])))
    # calculate mean parameters
    theta_Rs = (Nj_Rs + aR) / (Nc_Rs + aR + bR)
    theta_Ss = (Nj_Ss + aS) / (Nc_Ss + aS + bS)
    prior_R = float(Nc_R + alphaR) / (Nc_R + Nc_S + alphaR+alphaS)
    prior_S = float(Nc_S + alphaS) / (Nc_R + Nc_S + alphaS+alphaR)
    # get distribution over thetas
    newBs_R = Nc_Rs - Nj_Rs + bR
    newBs_S = Nc_Ss - Nj_Ss + bS
    newAs_R = Nj_Rs + aR
    newAs_S = Nj_Ss + aS      
    if distOpt==1:      # return distribution parameters, too
        return theta_Rs[0], theta_Ss[0], prior_R, prior_S, newAs_R[0], newAs_S[0], newBs_R[0], newBs_S[0]
    else:
        return theta_Rs[0], theta_Ss[0], prior_R, prior_S
    
    
def predictBernoulliNaiveBayes(testSet, theta_Rs, theta_Ss, prior_R, prior_S):
    """
    predict, probR, probS = predictBernoulliNaiveBayes(testSet, theta_Rs, theta_Ss, prior_R, prior_S)#
    
    Input:
    - 
    
    Output:
    - predict: array of 1's and 0's indicating class prediction
    - probR: array holding the probability of the R class for every example
    - probS: array holding the probability of the S class for every example
    
    Written by Kate Niehaus, 7-Apr-2014
    """
    numFeatures = np.shape(testSet)[1]
    probR = np.zeros(((np.shape(testSet)[0]),1))
    probS = np.zeros(((np.shape(testSet)[0]),1))
    predict = np.zeros(((np.shape(testSet)[0]),1))
    for i in range(np.shape(testSet)[0]):       # for all examples
        newEx = testSet[i,:]
        prodR = 0
        prodS = 0
        for j in range(numFeatures):        # for all features
            if newEx[j]==1:     # if class R
                prodR = prodR + np.log(theta_Rs[j])
                prodS = prodS + np.log(theta_Ss[j])
            elif newEx[j]==0:       # if class S
                prodR = prodR + np.log((1-theta_Rs[j]))
                prodS = prodS + np.log((1-theta_Ss[j]))
        # get temporary max to use log-sum-exp method
        probR_temp = prodR + np.log(prior_R)
        probS_temp = prodS + np.log(prior_S)
        pred = max(probR_temp, probS_temp)  
        evidence = np.log(np.exp(probR_temp-pred) + np.exp(probS_temp-pred)) + pred
        # calculate probabilities
        probR[i] = np.exp(probR_temp - evidence)
        probS[i] = np.exp(probS_temp - evidence)       
        #probS[i] = prodS*prior_S / (prodR*prior_R + prodS*prior_S)
        if probR[i] > probS[i]:
            predict[i] = 1    
    return predict, probR, probS
    
    
    
    
def returnProbClassGivenParam_sampling(trainSet, trainLabels, ind):
    """
    points, trueEst = returnProbClassGivenParam_sampling(trainSet, trainLabels, ind)
    """
    a_paramPriorR = 1
    b_paramPriorR = 0.25
    a_paramPriorS = 0.25
    b_paramPriorS = 1
    a_paramPriorClassR = 0.5
    b_paramPriorClassR = 0.5
    dataSliceR = trainSet[np.where(trainLabels==1)[0],ind]
    dataSliceS = trainSet[np.where(trainLabels==0)[0],ind]
    # parameters for resistance class beta
    a_probParamGivenClassR = sum(dataSliceR) + a_paramPriorR
    b_probParamGivenClassR = len(dataSliceR) - sum(dataSliceR) + b_paramPriorR
    # parameters for sus class beta
    a_probParamGivenClassS = sum(dataSliceS) + a_paramPriorS
    b_probParamGivenClassS = len(dataSliceS) - sum(dataSliceS) + b_paramPriorS
    # parameters for prob of resistance beta
    a_probClassR = len(dataSliceR) + a_paramPriorClassR
    b_probClassR = len(dataSliceS) + b_paramPriorClassR
    # Sample from distributions
    points = []
    numDraws = 1000
    betaR = np.random.beta(a_probParamGivenClassR, b_probParamGivenClassR, numDraws)
    betaClass = np.random.beta(a_probClassR, b_probClassR, numDraws)
    betaS = np.random.beta(a_probParamGivenClassS, b_probParamGivenClassS, numDraws)
    points = ((betaR*betaClass)/(betaR*betaClass+betaS*betaClass))
    # Trying with MC
#    betaR = stats.beta(a_probParamGivenClassR, b_probParamGivenClassR).rvs(numDraws)
#    betaClass = stats.beta(a_probClassR, b_probClassR).rvs(numDraws)
#    betaS = stats.beta(a_probParamGivenClassS, b_probParamGivenClassS).rvs(numDraws)
#    p = betaR*betaClass
#    dens_p = sm.nonparametric.KDEUnivariate(p)
#    dens_p.fit()
#    points = runMC(dens_p)
    # get "true" estimate of point estimate
    trueEstNum = (sum(dataSliceR)/float(len(dataSliceR)))*(len(dataSliceR)/float(len(trainSet)))
    trueEstDenom = sum(trainSet[:,ind])/float(len(trainSet))
    trueEst = trueEstNum/float(trueEstDenom)
    return points, trueEst
    


def runMC(dens_p):
    """
    This function performs MCMC for the proposal function N(xi, 0.5) and with 
    p(x) ~ dens_p
    
    """
    numSamples = 10000
    x_is = np.zeros((numSamples,1))
    x_is[0] = 0.5
    propStd = 0.25
    for i in range(numSamples-1):
        #print(i)
        u = np.random.uniform(low=0, high=1, size=1)
        x_star = np.random.normal(x_is[i], propStd, size=1)
        criterionTop = dens_p.evaluate(x_star)[0]*stats.norm(x_star, propStd).pdf(x_is[i])
        criterionBottom = dens_p.evaluate(x_is[i])[0]*stats.norm(x_is[i], propStd).pdf(x_star)
        criterion = criterionTop/float(criterionBottom)
        finalCriterion = np.min([criterion, 1])
        if u < finalCriterion:
            x_is[i+1] = x_star
        else:
            x_is[i+1] = x_is[i]
    return x_is

 
    
    
    
def BernoulliNaiveBayes(trainSet, trainLabels, testSet, testLabels, DAind):
    """
    
    
    """
    # fit model in train set
    theta_Rs, theta_Ss, prior_R, prior_S = fitBernoulliNaiveBayes(trainSet, trainLabels, DAind, 0)
    # test model in test set
    predict, probR, probS = predictBernoulliNaiveBayes(testSet, theta_Rs, theta_Ss, prior_R, prior_S)
    
    results = calcPerf(testLabels, predict, 1)
    return results, predict, probR, probS, theta_Rs, theta_Ss
    
    
    
def plotBetaDist(a,b):
    """
    plotBetaDist(a,b)
    
    This function will sample from the beta distribution governed by a and b
    100 times and produce a histogram of these results    
    
    Input:
    - a, b parameters for a beta distribution
    
    Written by Kate Niehaus, 7-Apr-2014
    """
    betaDist = []
    for k in range(100000):
        betaDist.append(random.betavariate(a, b))
    plt.figure()
    plt.hist(betaDist)
    plt.xlim((0,1))
    #return betaDist
    
def returnBetaDist(a,b):
    """
    plotBetaDist(a,b)
    
    This function will sample from the beta distribution governed by a and b
    100 times and produce a histogram of these results    
    
    Input:
    - a, b parameters for a beta distribution
    
    Written by Kate Niehaus, 7-Apr-2014
    """
    betaDist = []
    for k in range(100000):
        betaDist.append(random.betavariate(a, b))
    return betaDist
    
    
    
#######################################################################################
#
### Clustering-related
#
#######################################################################################
    
    
   
      
  

def matchClusterAssigns(assigns):
    """
    matched_assigns = matchClusterAssigns(assigns)
    
    Input: 
    - assigns: (num examples x num iterations)
    
    Output
    - matched_assigns: assignments matched so that classes correspond
    
    
    Written by KN, 27-Jan-2016    
    """
    N, numIts = np.shape(assigns)
    matched_assigns = np.zeros_like(assigns)
    adjRand = []
    # go through columns
    g1 = assigns[:,0]
    for i in range(numIts-1):
        #g1 = assigns[:,i]
        g2 = assigns[:,i+1]
        confs = sk.metrics.confusion_matrix(g1,g2)
        groupNums1 = np.sort(np.unique(g1))
        groupNums2 = np.sort(np.unique(g2))[1:]  # limit range so that leave out the zeros
        inds = np.argmax(confs,0)[1:]   # limit range so that leave out the zeros
        newAssigns = groupNums1[inds]
        # 
        clean_group2 = np.zeros_like((g2))
        for j in range(len(groupNums2)):
            clean_group2[g2==groupNums2[j]] = newAssigns[j]
            
        adjR1 = sk.metrics.adjusted_rand_score(g1,g2)
        adjR2 = sk.metrics.adjusted_rand_score(g1, clean_group2)
        if adjR1==adjR2:
            print('mismatch after reassignment')
        adjRand.append(adjR2)
        
        matched_assigns[:,i+1] = clean_group2
    return matched_assigns, adjRand
    
    
    
    
def jitterClustersAndEvaluate(dataMatrix, numClusters, prop=0.9, method='hierarchical', numSubSamples=10):
    """
    matched_assigns, adjRand = jitterClustersAndEvaluate(dataMatrix, numClusters, prop, method)
    
    This function subsamples a proportion of the input patients and conducts 
    the clustering multiple times to see how consistent it is
    
    Input 
    - dataMatrix: assumed to be post all preprocessing (e.g. no normalization
        or logs are taken)
        (numPts x numFeats)
    - numClusters: numerical number set
    - prop: proportion of patients to subsample
    - method: method of clustering
        'hierarchical' is all that is implemented so far
    
    Written by KN, 27-Jan-2016
    """
    N, M = np.shape(dataMatrix)
    nTrain = int(np.round(prop*N))
    
    # initialize
    assigns = np.zeros((N,numSubSamples))    
    
    # subsample
    for i in range(numSubSamples):
        # Split data
        indices = np.random.permutation(N)
        trainInd, testInd = indices[:nTrain], indices[nTrain:]
        # get clusters
        newClusterObj= AgglomerativeClustering(n_clusters=numClusters, affinity='euclidean', linkage='ward')
        trainSet = dataMatrix[trainInd,:]
        trainSet_norm = normalizeByCols(trainSet)
        newClusters = newClusterObj.fit_predict(trainSet_norm)
        assigns[trainInd,i] = newClusters+1
        
    # get all assignments converted to the same labels
    matched_assigns, adjRand = matchClusterAssigns(assigns)    

    return matched_assigns[:,1:], adjRand

    

def plotDendrogram(fullMatrix, thresh=100, colors='na'):
    """
    This function will plot a dendrogram of the input data
    
    """
    fs=15
    matplotlib.rc('xtick', labelsize=fs)
    matplotlib.rc('ytick', labelsize=fs)
    matplotlib.rc('font', size=fs)
    matplotlib.rcParams.update({'font.size':fs})
    plt.figure(figsize=[18,6])
    newLinkage = linkage(fullMatrix, method='ward', metric='euclidean')     # create the actual clustering
    if colors=='na':
        dendrogramDict = dendrogram(newLinkage, color_threshold=thresh, leaf_rotation=90, leaf_font_size=4)      # draw the dendrogram
    else:
        dendrogramDict = dendrogram(newLinkage, color_threshold=thresh, leaf_rotation=90, leaf_font_size=4, link_color_func = lambda k: colors[k])      # draw the dendrogram
    plt.xlabel('Patients'); plt.ylabel('Distance')
    
    
    
    
def plotClusterTrajectories(numClusters, featMat, xvals, figsize=[10,10], ymin = 0, ymax=100, clustOrder=[], ylabel='Anomaly score', xlabel='Time from diagnosis (years)'):
    """
    This function will plot the trjaectories of the patients, according to 
    their clustered results
    
    Written by KN, 3-May-2016
    """
    newClusterObj= AgglomerativeClustering(n_clusters=numClusters, affinity='euclidean', linkage='ward')
    newClusters = newClusterObj.fit_predict(featMat)
    
    #xvals = np.linspace(-.5,FU, numFeats)
    numRows = int(np.ceil(np.sqrt(numClusters)))
    
    
    # get cluster order
    if len(clustOrder)<1:
        clustOrder = range(numClusters)
    
    plt.figure(figsize=figsize)
    counter=1
    for j in clustOrder:
        ptFeatureInds = np.where(newClusters==j)[0]
        plt.subplot(numRows,numRows,counter)
        for i in ptFeatureInds:
            plt.plot(xvals, featMat[i,:])
        plt.ylim([ymin,ymax])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if len(clustOrder)<1:
            plt.title('Group {0} (N={1})'.format(j, len(ptFeatureInds)))
        else:
            plt.title('N={0}'.format(len(ptFeatureInds)))
        plt.vlines(0,0, ymax, linestyle='--', color='FireBrick')
        counter+=1
    plt.tight_layout()
    
    return newClusters    
    
    
    
    
def plotClusterTrajectoriesBySubfeats(numClusters, featMat, xvals, numFeatsInSet, featLabels, opt='indiv', fs=12, figsize=[10,10], ymax=100, clustOrder=[]):
    """
    This function will plot the trjaectories of the patients, according to 
    their clustered results, but split into sets of input feature types (e.g. 
    different labs)
    
    opt: 'indiv' is each patient individually; 'mean' is the average of scores
    
    Written by KN, 13-Jul-2016
    """
    newClusterObj= AgglomerativeClustering(n_clusters=numClusters, affinity='euclidean', linkage='ward')
    newClusters = newClusterObj.fit_predict(featMat)
    
    #xvals = np.linspace(-.5,FU, numFeats)
    numRows = numClusters
    numCols = np.shape(featMat)[1]/np.float(numFeatsInSet)
    
    
    # get cluster order
    if len(clustOrder)<1:
        clustOrder = range(numClusters)
    
    plt.figure(figsize=figsize)
    matplotlib.rc('xtick', labelsize=fs)
    matplotlib.rc('ytick', labelsize=fs)
    matplotlib.rc('font', size=fs)
    matplotlib.rcParams.update({'font.size':fs})
    counter=1
    for j in clustOrder:
        ptFeatureInds = np.where(newClusters==j)[0]
        for k in range(int(numCols)):
            plt.subplot(numRows,numCols,counter)
            if opt=='indiv':
                for i in ptFeatureInds:
                    plt.plot(xvals, featMat[i,k*numFeatsInSet:(k+1)*numFeatsInSet])
            elif opt=='mean':
                meanScore = np.mean(featMat[ptFeatureInds,k*numFeatsInSet:(k+1)*numFeatsInSet], axis=0)
                plt.plot(xvals, meanScore, color='FireBrick')
                stdScore = np.std(featMat[ptFeatureInds,k*numFeatsInSet:(k+1)*numFeatsInSet], axis=0)
                seScore = stdScore/np.sqrt(len(ptFeatureInds))
                plt.fill_between(xvals, meanScore-seScore, meanScore+seScore, alpha=0.5, color='Indigo')
            plt.ylim([0,ymax])
            plt.xlabel('Time from diagnosis (years)')
            plt.ylabel('{0} anomaly score'.format(featLabels[k]))
            if len(clustOrder)<1:
                plt.title('Group {0} (N={1})'.format(j, len(ptFeatureInds)))
            else:
                plt.title('N={0}'.format(len(ptFeatureInds)))
            plt.vlines(0,0, ymax, linestyle='--', color='FireBrick')
            counter+=1
    plt.tight_layout()
    
    return newClusters 
    
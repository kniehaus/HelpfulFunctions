# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:47:16 2016

@author: trin2441
"""


import csv
import numpy as np
import colorsys
import matplotlib
import matplotlib.pylab as plt
import pandas as pd

#######################################################################################
#
### Generic
#
#######################################################################################

def readcsv(fileName, rowSkip=0, noHeader=False, delim=',', asArray=True):
    """ 
    Reads in a .csv file path for a set of data and returns the contents as 
    a header list and (array) of the data
    
    NOTE: assumes .csv file
    
    ----------
    
    Input
    
    fileName: string
        File path for data file

    rowSkip: int (default=0)
        Number of rows to skip at the start of the file 
        
    noHeader: boolean (default=False)
        Whether there is not a header (=True) or that there is a header (=False)
        
    delim: string (default=',')
        Delimiter for file
        
    asArray: boolean (default=True)
        Whether to convert data contents into an array (=True) or return as 
        list (=False)
        
    ----------
    
    Output
    
    header: list
        List of strings describing column contents
        
    data: array or list
        Numpy array or list (depending on input argument asArray) of file's
        data contents
    
    
    Written by Kate Niehaus, 26-May-2014
        
    """
    f_reader = csv.reader(open(fileName, 'rU'), delimiter=delim)
    for i in range(rowSkip):
        next(f_reader)
    # Get header line
    if noHeader==True:
        header = [] 
    else:
        header = next(f_reader)
    # Initialize
    data = []
    for entry in f_reader:
        data.append(entry)
    if asArray==True:
        data = np.array(data)
    return list(header), data

    
    

def readcsvAndReturnDF_generic(fileName, indexOpt='default', **kwargs):
    """
    Reads in a .csv file path for a set of data and returns the contents as 
    a pandas dataframe
    
    NOTE: assumes .csv file
    
    ----------
    
    Input
    
    fileName: string
        File path for data file

    indexOpt: string (default='default')
        'default' means to use the first column of the input data
        as the index.  Anything else will return a DF without an index
        
    **kwargs as keyword args for readcsv
        
    ----------
    
    Output
    
    df: pandas dataframe
        Dataframe of data contained in file, with first line as header and 
        first column as index (using default settings)
    
    
    Written by KN, 17-June-2016    
    
    """    
    header, data = readcsv(fileName, **kwargs)
    if indexOpt=='default':
        df = pd.DataFrame(data, columns=header, index = data[:,0])
    else:
        df = pd.DataFrame(data, columns=header)
    return df
    
    
    
    
def writecsv_generic(saveFile, header, data):
    """
    Writes a .csv file for input data with the given header
    
    NOTE: assumes data can be formatted into an array and everything is easy
    
    ----------
    
    Input
    
    saveFile: string
        File path for file to save

    header: list
        List of column names
        
    data: list of lists or array
        Data to save
        
    ----------
    
    Output
    
    (None; writes file)
    

    Written by KN, 28-Aug-2016    
    
    """
    data = np.array(data)
    # save to csv
    with open(saveFile, 'wb') as csvFile:
        newWriter = csv.writer(csvFile, delimiter=',')
        newWriter.writerow(header)
        newWriter.writerows(data)
    



def returnColors(n, offset=0, rand=False):
    """
    Returns a color palette given an input number of colors
    
    default: (x, .65, 1)
    See: https://en.wikipedia.org/wiki/Web_colors
    
    ----------
    
    Input
    
    n: int
        Number of colors to return
        
    offset: float (default=0)
        To change default hue setting
        
    rand: Boolean (default=False)
        Whether to return color choices in random order (=True) or in color-
        wheel order (=False)
        
    ----------
    
    Output
    
    colors: list
        List of RGB color codes
    

    Written by KN, ~April 2016 
    
    """
    hues = np.linspace(15+offset, 375+offset, n+1)
    colors = [colorsys.hls_to_rgb(hue/360, .55, .35) for hue in hues]
    if rand==True:
        colors = np.random.permutation(colors)
    return colors
    
    
    
    
def resetGraphing(opt='default'):
    """
    Sets the plotting to either 'default' or 'ggplot'

    """    
    if opt=='ggplot':
        matplotlib.style.use('ggplot')
    elif opt=='default':
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    
    
def createDict(list1, list2):
    """ 
    Returns a dictionary linking everything in list1 to the corresponding 
    item in list2
    
    ----------
    
    Input
    
    list1: list
        List of items
        
    list2: list
        List of items of same length as list1 and with corresponding items
        
    ----------
    
    Output
    
    newDict: dictionary
        Dictionary linking two lists
   
    """
    newDict = {}
    for line1, line2 in zip(list1, list2):
        newDict[line1]=line2
    return newDict
    
    
    
    
def scatterWithNoise(x1, y1):
    """
    Scatter plot of x1 v y1, with random noise added to each

    Written by KN, 25-Apr-2016
    """    
    colors=returnColors(5)
    N = len(x1)
    new_x = x1.reshape(N,1) + np.random.normal(loc=0, scale=0.05, size=(N,1))
    new_y = y1.reshape(N,1) + np.random.normal(loc=0, scale=0.05, size=(N,1))
    
    plt.plot(new_x, new_y, 'o', color = colors[4], alpha=0.5)
    
    
    
   
def nanUnique(data):
    """
    Returns unique items in list, even if NANs exist
    """
    data_c = data[~np.isnan(data)]
    uniq = np.unique(data_c)    
    return uniq
    
    
    
    
def nanHist(data, xlabel='', ylabel='', title='', normed=False, logOpt=0, **kwargs):
    """
    Plots histogram with nan's removed
    
    ----------
    
    Input
    
    data: array
        Data to plot
    
    xlabel, ylabel, title: strings (default='')
        Labels for x and y axes and title, respectively
        
    normed: Boolean (default=False)
        Whether to normalize data (=True) or keep raw counts (=False)
        
    logOpt: int (default=0)
        Whether to take log of data (=1) or not (=0)
        
    **kwargs as keyword arguments for matplotlib hist plotting
    
    ----------
    
    Output
    
    [None; plots histogram]
        
        
    """
    data= np.array(data)
    data_c = data[~np.isnan(data)]
    if logOpt==1:
        try:
            data_c = np.log(data_c)
        except:
            data_c = np.log(data_c+.0001)
    plt.hist(data_c, normed=normed, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    
    
def setFontsize(fs):
    """
    Resets fontsize to fs everywhere
    
    """
    matplotlib.rc('xtick', labelsize=fs)
    matplotlib.rc('ytick', labelsize=fs)
    matplotlib.rc('font', size=fs)
    matplotlib.rcParams.update({'font.size':fs})
    font = {'size'   : fs}
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'axes.titlesize':fs})

    

def timeDiffDays(time1, time2):
    """
    Returns the difference of time1-time2 (both are datetime objects) in terms of days
    
    """
    return (time1-time2).days/365.25
    
    
    
    
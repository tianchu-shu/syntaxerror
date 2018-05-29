'''
Amir Akhter Kazi
Functions for Data Exploration and Visualization 
25th May 2018
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import matplotlib

# For Jupyter Notebook
# %matplotlib inline




def basic_exploring (dataframe):
    '''
    Given a dataframe, the function does some basic exploring
        by printing the description and information of the dataset
        
    input:
        dataframe
    '''
    print ('DESCRIBING DATASET: \n \n', dataframe.describe(), '\n \n') 
    print ('DATASET INFORMATION \n') 
    print (df.info(), '\n \n \n')
    print ('DATASET HEAD \n')
    print (df.head(), '\n \n \n')
    print ('DATASET TAIL \n')
    print (df.tail(), '\n \n \n')
    
    for column in df.columns:
        plotting_curves (df, column)
        plt.show()


def draw_correlation_matrix (df, title):
    '''
    Creates a heatmap that shows the correlations between the different variables in a dataframe.
    
    Input:
        df: a dataframe
        title: name of the correlation_matrix
        
    Return:
        Outputs a heatmatrix showing correlations
    
    Code based on: https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas
    '''
    ax = plt.axes()
    corr = df.corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, ax = ax) 
    ax.set_title (title)


def plotting_curves (dataframe, feature):
    '''
    Given a dataframe, a column name, 
        displays a plot of that dataframe column distribution.
        
    Input:
        dataframe
        feature: column name (string)
        
    Return:
        displays a distribution of that variable
        
    Inspired by:
        https://seaborn.pydata.org/generated/seaborn.distplot.html
    '''
    title = feature + ' Graph'
    ax = sns.distplot(dataframe[feature])
    ax.set_title(title)


def splitting_across_variable(dataframe, feature):
    '''
    For a given dataframe and feature,
        it splits the database as per the values of the feature,
        and gives aggregrate information for that feature and
        how it relates to other features
        
    Input:
        dataframe
        
    Output: 
        dataframe
    '''
    return dataframe.groupby([feature]).mean().transpose()


def value_counter (dataframe, feature):
    '''
    Given a dataframe, and feature,
        gives the number of occurrences of each value in that feature column
        
    Input:
        dataframe
        feature: string
    
    Output: dataframe
    '''
    return dataframe[feature].value_counts()

def comparing_two_features(dataframe, feature_1, feature_2):
    '''
    Given a dataframe, and two features,
        does a cross tab and shows values across those two corresponding features
        
    Input:
        dataframe
        feature_1, feature_2: string
        
    Output: 
        dataframe
        
    '''
    return pd.crosstab(dataframe[feature_1], dataframe[feature_2])


def plotting_two_feature_comparison(dataframe, x_feature, y_feature):
    '''
    Given a dataframe and two features,
        plots a graph across those two features
        
    Input:
        dataframe
        x_feature, y_feature: string
        
    Output:
        matplotlib.axes._subplots.AxesSubplot
    '''
    return dataframe[[x_feature, y_feature]].groupby(x_feature).sum().plot()


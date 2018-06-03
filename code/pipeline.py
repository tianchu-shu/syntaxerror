import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# Read Data
def Read_data(filename):
    '''
    Input:
        filename (string): name of the file with the words
    Returns:
        pandas dataframe
    '''
    if '.csv' in filename:
        df = pd.read_csv(filename, index_col = 0)
        print(filename + " is sucessfully loaded")
        return df

    elif '.xls' in filename or 'xlsx' in filename:
        df = pd.read_excel(filename, index_col = 0)
        print(filename + " is sucessfully loaded")
        return df

    elif '.json' in filename:
        df = pd.read_json(filename, index_col=0)
        print(filename + " is sucessfully loaded")
        return df
    
    else:
        print("This func takes .csv/.xls/.xlsx/ .json files" )


# Explore Data

def exploring_overview(df):
    '''
    This function gives a high level view of the dataset.
    It states the attributes which exists, and provides information 
        about the dataset.
    '''
    list_of_columns = df.columns
    print ('LIST OF COLUMNS: ')
    print (list_of_columns)
    print ()
    
    print ('NUMBER OF ROWS ARE: ', df.shape[0])
    print ( 'NUMBER OF COLUMNS ARE: ', df.shape[1])
    print ()
    
    print ('SOME BASIC INFORMATION ABOUT THE ATTRIBUTES: ')
    print (df.info())
    print ()
    
    print ('DESCRIPTION OF THE DATASET:')
    print (df.describe())

def grouping_by_feature (feature, df):
    '''
    For a given feature, it separates the data for the different values of that feature.
        example: for a variable 'SeriousDlqin2yrs', where two values exist i.e. 0 and 1,
            the function separates the other variables and gives a grouped description of 
            how other variable statistics vary with this feature's division
    '''
    print (df.groupby(feature).mean().transpose())

def comparing_across_two_features (feature1, feature2, df):
    '''
    For any two features, it compares the variation in the data across those two features
    '''
    first_entry = 'df.' + feature1
    second_entry = 'df.' + feature2
    
    try:
        pd.crosstab(first_entry, second_entry)
    except:
        print ()

def summing_nulls_in_dataset (df):
    '''
    Sums the null values in every attribute of the dataset, and states the null values in each 
    '''
    print (df.isnull().sum())

def counting_in_a_variable(feature, df):
    '''
    For a given feature and dataframe,
        prints the total count of each category of that feature.
    '''
    print (df[feature].value_counts())

def counting_uniques(df):
    '''
    For the given dataframe, gives a sum of the unique values in each feature.
    Then prints a plot bar to represent that.
    '''
    print (df.nunique())
    print (df.nunique().plot.bar())


# Pre-Process Data
def Detect_missing_value(df):
    '''
    Find out the columns have missing values
    Input:
        pandas dataframe
 
    Returns:
        a list of those column names
    '''
    rv = []
    for col in df.columns:
        if df[col].count() < df.shape[0]:
            rv.append(col)
            print(col, "has missing values.")
    return rv


def Fill_in(df, cols, method="mean"):
    '''
    Filling in missing values with "mean" or "median"
    
    Inputs:
        pandas dataframe
        a list of those column names
        method (string): mean or median
 
    Returns:
        pandas dataframe
    '''
    for col in cols:
        if method =='mean':
            val = df[col].mean()
        if method =='median':
            val = df[col].median()
        df[col] = df[col].fillna(val, inplace = True)
        print ('Filling missing value for {} using {}'.format(col, method))
    return df


def Imputer(df, cols):
    # make copy of the original data
    copy_df = df.copy()

    # make new columns indicating what will be imputed
    cols_with_missing = (col for col in copy_df.columns if copy_df[col].isnull().any())
    for col in cols_with_missing:
        copy_df[col + '_was_missing'] = copy_df[col].isnull()

    # imputation
    imputer = Imputer()
    imputed_df = pd.DataFrame(imputer.fit_transform(copy_df))
    imputed_df.columns = copy_df.columns

    # rounding numbers
    for cols in columns:
        imputed_df[cols].round(0)

    return imputed_df


# HANDLE OUTLIERS

def remove_outlier(df, col_name):
    '''
    Input:
        pandas dataframe
        col_name (string): name of the column
    Returns:
        pandas dataframe without outliners
    '''
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3 - q1 #Interquartile range
    fence_low  = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
    num = df.shape[0] - df_out.shape[0]

    print(num, "rows of outliners of " + col_name + " are dropped.")
    return df_out


# SETTING THE CEILINGS & BUCKETING

def ceiling(x, cap_max):
    if x > cap_max:
        return cap_max
    else:
        return x

def flooring(x, cap_min):
    if x > cap_min:
        return cap_min
    else:
        return x

def set_boundaries(df,columns):

    for col in columns:
        upper_outlier = df[col].quantile(.999)
        df[col] = df[col].apply(lambda x: ceiling(x, upper_outlier))
        lower_outlier = df[col].quantile(.001)
        df[col] = df[col].apply(lambda x: flooring(x, lower_outlier))

    return None

def into_buckets(df, columns, bins = 0, quant = 0):
    # q=4 for quartiles, q=10 for deciles
    for col in columns:

        if bins != 0:
            df[col[:len(df.columns)]+'_buckets'] = pd.cut(df[col], bins=bins)
        if quant != 0:
            df[col[:len(df.columns)]+'_buckets'] = pd.qcut(df[col], q=quant)

    return df


# CREATE DUMMIES

def create_dummy(df, columns):
    for col in columns:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)

    return df

# Visualize

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


def plotting_curve (dataframe, column, title):
    '''
    Given a dataframe, a column name, and a title,
        displays a plot of that dataframe column distribution.
        
    Input:
        dataframe
        column: column name (string)
        title: string
        
    Return:
        displays a distribution of that variable
        
    Inspired by:
        https://seaborn.pydata.org/generated/seaborn.distplot.html
    '''
    try:
        ax = sns.distplot(dataframe[column])
        ax.set_title(title)
        plt.show()
    except:
        pass


def making_pie (df, feature):
    '''
    Gives a pie plot of data in any feature
    '''
    try: 
        df.groupby([feature]).size().plot.pie()
    except:
        pass


def plotting_bar (df, feature):
    '''
    Plots a bar graph based on the size of each category of that feature
    '''
    df.groupby([feature]).size().plot.bar()


def plotting_top_10_bar_plot  (df, feature):
    '''
    Plots a bar plot for the top 10 common values of a given feature
    '''
    df.groupby([feature]).size().sort_values().iloc[-10:].plot.bar()


# Classifiers

def split_data(train, test, y):
    '''
    Split the data into training and testing set
    
    And save them to run try different models
    
    y(str): the name of the target y
    '''
    x_test = test[indepv] 
    x_train = train[indepv]
    y_test = test[y]
    y_train = train[y]
    return x_train, x_test, y_train, y_test



def Classifier(model, num, x_train, y_train, x_test):
    if model == 'LR':
        clf = LogisticRegression('l2', C=num)
    elif model == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=num)
    elif model == 'DT':
        clf = DecisionTreeClassifier(max_depth=num)
    elif model == 'RF':
        clf = RandomForestClassifier(max_depth=num)
    elif model == 'BAG':
        clf = BaggingClassifier(max_samples=num, bootstrap=True, random_state=0)
    elif model == 'BOOST':
        clf = GradientBoostingClassifier(max_depth=num)
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return y_pred

# Evaluation

def confusion_matrix(self, y_test, y_pred):

    result_matrix = confusion_matrix(y_test, y_pred)

    return result_matrix

def precision_table(self, y_test, y_pred):

    table = classification_report(y_test, y_pred)

    return table

def portfolio_detail(self, test_in, y_pred):

    y_port = test_in[y_pred == 1]

    return y_port.shape[0], y_port.mean()


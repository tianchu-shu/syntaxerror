from final_default_grids import *
from datetime import *

import sys
from IPython.display import display
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

NOTEBOOK = True

GRID = SMALL_GRID
MODELS_TO_RUN = ['RF','DT','KNN', 'NB', 'ET']


START = "2010-01-01"
END = "2016-04-30"
WINDOWS = [6, 12]
UPDATE = 12


def load_from_file(dataframe_string, row_num = None):
    '''
    Function reads the .csv file into a panda dataframe.
    Inputs:
    dataframe_string: String for file name
    filetype: string for file_type
    row_num: integer, number of rows to read if excel or csv
    Outputs:
    df = A panda dataframe
    '''
    if '.csv' in dataframe_string:
        df = pd.read_csv(dataframe_string, nrows = row_num)
    elif '.xls' in dataframe_string:
        df = pd.read_excel(dataframe_string, nrows = row_num)
    elif '.json' in dataframe_string:
        df = pd.read_json(dataframe_string)
    
    print("Loaded" + dataframe_string)
    
    return df



def load_from_db(table_list):
    df_dict = {}
    conn = connection.Connect()
    for x in table_list:
        file_name = x + "_df"
        df_dict[file_name] = connection.return_df('table', x)
        print("Loaded" + file_name)
    
    return df_dict





########################### PRE-PROCESSING ###########################

# CONVERT INTO DATETIME FORMAT
def to_datetime(df, cols, index=-8, format="%Y%m%d"):
    for col in cols:
        df[col] = df[col].apply(lambda x: str(x)[:index])
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = pd.to_datetime(df[col], format=format)
        
    return df



# CUTTING & MERGING DATAFRAME WITH DATETIME RANGE 
def restrain_datetime(df, date_col='arrest_date', from_date=(2010,1,1), to_date=(2015,12,31)):
    df.index = df[date_col]
    df = df.sort_index()
    df = df[datetime(*from_date):datetime(*to_date)]
    df = df.reset_index(drop=True)

    return df



def checking_for_nulls(dataframe):
    '''
    Given a dataframe, checks for columns which have NaN or Nulls,
        and returns a list with the name of those features which have NaN or Nulls.
        
    Input:
        dataframe
        
    Output:
        features_with_nulls: list of strings
    '''
    features = dataframe.columns
    features_with_nulls = []

    for column in df.columns:    
        if df[column].isnull().sum() > 0:
            features_with_nulls.append(column)
    
    return features_with_nulls



# FILL IN MISSING VALUES WITH SELECTED METHODS FOR EACH TYPE 
def fill_missing(df, method1="missing", method2="median"):

    for col in df:
        if df[col].dtype == 'object' and method1 == "missing":
            df[col].fillna("Missing")
        elif df[col].dtype == 'object' and method1 != "missing":
            try:
                df[col].fillna(df[col].mode()[0], inplace=True)
            except:
                df[col].fillna("Missing")
        elif df[col].dtype == 'int' or df[col].dtype == 'float':
            if method == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            if method == "median":
                df[col].fillna(df[col].median(), inplace=True)

    return df



def outlier(df, variable):
    '''
    Locate outliers in given column and eliminate them.
    Inputs:
    df: A pandas dataframe
    variable: target column name, string
    Outputs:
    changed_df: A pandas dataframe without the outliers
    '''
    low_out = df[variable].quantile(0.005)
    high_out = df[variable].quantile(0.995)
    df_changed = df.loc[(df[variable] > low_out) & (df[variable] < high_out)]

    number_removed = df.shape[0] - df_changed.shape[0]
    print("Removed" + str(number_removed) + " outliers from " + variable)

    return df_changed



# CHANGE IN VARS FOR EACH
def is_changed(df, id_num='mni_no', var='mar_stat'):
    df['{}_changed'.format(var)] = (df[var]!=df[var].shift()) | (df[id_num]!=df[id_num].shift())

    return df
    


# CUM SUM OF VARS
def cum_vals(df, id_num='mni_no', var='re-enter-days'):

    df = df.sort_values(by=[id_num, var])
    df['{}_so_far'.format(var)] = df.groupby(id_num)[var].cumsum()

    return df



# CONVERT GIVEN VALS(val1, val2) TO BINARY VALS(0,1) 
def special_convert(df, cols, val1="t", val2="f"):
    for col in cols:
        df[col] = df[col].map({val1: 1, val2: 0})

    return df


def processing_drop(df, drop_list, target_quantifier, value):
    '''
    1) Drops all rows where the variables in the drop_list value where the target is less than, greater than, or equal to a value.
    Input:
    df: A panda dataframe
    drop_list: List of columns to act on
    maximum: The integer to drop if the value is greater than
    Outputs:
    df
    '''
    for variable in drop_list:
        if target_quantifier == 'equal':
            df = df[df[variable] == value]
        elif target_quantifier == 'greater':
            df = df[df[variable] >= value]
        elif target_quantifier == 'lesser':
            df = df[df[variable] <= value]
    
    return df


# ROUND VALUES OF GIVEN COLUMNS WITH SPECIFIED PLACE
def round_to(df, cols=None, digit=0):

    if not cols:
        cols = df.columns

    for col in cols:
        df[cols].round(digit)

    print('given columns are rounded to {}'.format(digit))

    return df


def bin_gen(df, variable, label, fix_value):
    '''
    Create a bin column for a given variable, derived by using the 
    description of the column to determine the min, 25, 50, 75 and max
    of the column. Then categorize each value in the original variable's
    column in the new column, labeled binned_<variable>, with 1,2,3,4
    Ranging from min to max
    Inputs:
    df: A panda dataframe
    variable: A string, which is a column in df
    label: A string
    fix_value: Either prefix or suffix
    Outputs:
    df: A panda dataframe
    '''
    variable_min = df[variable].min()
    variable_25 = df[variable].quantile(q = 0.25)
    variable_50 = df[variable].quantile(q = 0.50)
    variable_75 = df[variable].quantile(q = 0.75)
    variable_max = df[variable].max()
    
    bin = [variable_min, variable_25, variable_50, variable_75, variable_max]
    unique_values = len(set(bin))
    
    label_list = []
    iterator = 0
    for x in range(1, unique_values):
        iterator += 1
        label_list.append(iterator)
    
    if fix_value == 'prefix':
        bin_label = label + variable
    elif fix_value == 'suffix':
        bin_label = variable + label
    
    df[bin_label] = pd.cut(df[variable], bins = bin, include_lowest = True, labels = label_list)
    df.drop([variable], inplace = True, axis=1)
    
    df = dummy_variable(bin_label, df)
    
    return df


def dummy_variable(variable, df):
    '''
    Using the binned columns, replace them with dummy columns.
    Inputs:
    df: A panda dataframe
    variable: A list of column headings for binned variables
    Outputs:
    df:A panda dataframe
    '''
    dummy_df = pd.get_dummies(df[variable]).rename(columns = lambda x: str(variable)+ str(x))
    df = pd.concat([df, dummy_df], axis=1)
    df.drop([variable], inplace = True, axis=1)
    
    return df


# FILTER WITH FREQUENCY OF RE-ENTRY (PREV BOOKING DATE ~ NEXT BOOKING DATE)
def within_frame(df, id_num='dedupe_id', timestamp='booking_date', col_name='re-entry', duration=365):
    df = df.sort_values(by=[id_num, timestamp])
    df['{}'.format(col_name)] = df.groupby[id_num][timestamp].diff()
    df['{}'.format(col_name)] = df['{}'.format(col_name)].apply(lambda x: x.days)
    df['within_{}'.format(duration)] = np.where(df['{}'.format(col_name)]>duration, 1, 0)    
    
    return df



########################### BUILDING CLASSIFIERS & EVALUATION ###########################

# Based on Rayid's magicloops from here
# https://github.com/rayidghani/magicloops

def classifiers_loop(x_train, x_test, y_train, y_test, save=True):
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc', 'precision_5', 'accuracy_5', 'recall_5', 'f1_5',
                                                       'precision_10', 'accuracy_10', 'recall_10', 'f1_10',
                                                       'precision_20', 'accuracy_20', 'recall_20', 'f1_20',
                                                       'precision_30', 'accuracy_30', 'recall_30', 'f1_30',
                                                       'precision_50', 'accuracy_50', 'recall_50', 'f1_50'))
    for i, clf in enumerate([CLFS[x] for x in MODELS_TO_RUN]):
        print(MODELS_TO_RUN[i])
        params = GRID[MODELS_TO_RUN[i]]
        number = 1
        for p in ParameterGrid(params):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(x_train, y_train).predict_proba(x_test)[:,1]
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                print(p)
                accuracy_5, precision_5, recall_5, f1_5 = evals_at_k(y_test_sorted,y_pred_probs_sorted,5.0)
                accuracy_10, precision_10, recall_10, f1_10 = evals_at_k(y_test_sorted,y_pred_probs_sorted,10.0)
                accuracy_20, precision_20, recall_20, f1_20 = evals_at_k(y_test_sorted,y_pred_probs_sorted,20.0)
                accuracy_30, precision_30, recall_30, f1_30 = evals_at_k(y_test_sorted,y_pred_probs_sorted,30.0)
                accuracy_50, precision_50, recall_50, f1_50 = evals_at_k(y_test_sorted,y_pred_probs_sorted,50.0)
                results_df.loc[len(results_df)] = [MODELS_TO_RUN[i], clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       accuracy_5, precision_5, recall_5, f1_5,
                                                       accuracy_10, precision_10, recall_10, f1_10,
                                                       accuracy_20, precision_20, recall_20, f1_20,
                                                       accuracy_30, precision_30, recall_30, f1_30,
                                                       accuracy_50, precision_50, recall_50, f1_50]
                #plot_precision_recall_n(y_test,y_pred_probs,MODELS_TO_RUN[i]+str(number))
                #plot_roc(clf, y_test,y_pred_probs)
                number += 1
            except IndexError as e:
                    print(e)
                    continue

    if save:
        results_df.to_csv('results.csv', index=False)

    return results_df


def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def evals_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    accuracy = accuracy_score(y_true, preds_at_k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)

    return precision, accuracy, recall, f1


def plot_roc(name, probs, true, save=True):
    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name)
    pl.legend(loc="lower right")

    if save:
        plt.savefig('{}.png'.format(name[:15]))
    else:
        plt.show()

    return None


def plot_precision_recall_n(y_true, y_prob, model_name, save=True):
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)

    if save:
        plt.savefig('{}.png'.format(name))
    else:
        plt.show()

    return None



# BEST GRID
def best_grid(rdf, method = "auc-roc"):
    '''
    Iterate over the results and get the best parameters for each classifier
    and save the best_grid as a dictionary
    
    '''
    best = {}
    model = rdf.groupby("model_type")[method].nlargest(1)
    model = model.to_frame()
    model.reset_index(inplace = True)
    rows = list(model['level_1'])
    display(rdf.loc[rows].iloc[:,0:8])
    for row in rows:
        key = rdf.loc[row]["model_type"]
        v = rdf.loc[row]["parameters"]
        best[key] = v
    
    for k,arg in best.items():
        for key,val in arg.items():
            arg[key] = [val]

    return best 


# FEATURE IMPORTANCES 
def feature_importance(x_train, y_train, bestm, x="ET", k=10):
    '''
    Based on the best grid for each classifer, print out the 
    top k important features
    '''
    clf = CLFS[x]
    for p in ParameterGrid(bestm[x]):
        clf.set_params(**p)
    forest = clf.fit(x_train, y_train)
    indepv = list(x_train.columns)
    importances = forest.feature_importances_
    current_palette = sns.color_palette(sns.color_palette("hls", 8))
    indices = np.argsort(importances)[::-1]
    indices = indices[:k]

    # Print the feature ranking
    print("Feature ranking for %s" % (y_train.name))

    labels_arr = []
    for f in range(len(indices)):
        label = indepv[indices[f]]
        labels_arr.append(label)
        print("%d. %s (%f)" % (f+1, label, importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances for %s" % (y_train.name))
    plt.bar(range(len(indices)), importances[indices], align="center", tick_label = labels_arr, color= current_palette)
    plt.xlim([-1, len(indices)])
    plt.xticks(range(len(indices)),labels_arr, rotation = 'vertical')
    plt.show()




def plot_mult(models, x_train, x_test, y_train, y_test, bestm):
    '''
    Run model with the best given params on x and y
    and print out all the best models' on the same graph
    '''
    k = len(models)
    current_palette = sns.color_palette(sns.color_palette("Paired", k))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='#3498db')
    ax2 = ax1.twinx()
    ax2.set_ylabel('recall', color="#e74c3c")
    for index, clf in enumerate([CLFS[x] for x in models]):
        model_params = bestm[models[index]]
        for p in ParameterGrid(model_params):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(x_train, y_train).predict_proba(x_test)[:,1]
                
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_probs)
                precision_curve = precision_curve[:-1]
                recall_curve = recall_curve[:-1]
                pct_above_per_thresh = []
                number_scored = len(y_pred_probs)
                for value in pr_thresholds:
                    num_above_thresh = len(y_pred_probs[y_pred_probs>=value])
                    pct_above_thresh = num_above_thresh / float(number_scored)
                    pct_above_per_thresh.append(pct_above_thresh)
                pct_above_per_thresh = np.array(pct_above_per_thresh)
                
                ax1.plot(pct_above_per_thresh, precision_curve, c=current_palette[index])
                ax2.plot(pct_above_per_thresh, recall_curve, c=current_palette[index])                
            except IndexError as e:
                print(e)
                continue
    ax1.legend(models)
    plt.show()



def baseline(df, col):

    return df[col].sum() / df.shape[0]


# FILTER WITH FREQUENCY OF RE-ENTRY (PREV RELEASED DATE ~ NEXT BOOKING DATE)
def within_frame2(df, id='dedupe_id', col1='booking_date', col2='release_date', days=365, windows=[1]):
    
    # Change to datetime
    df[col1] = pd.to_datetime(df[col1])
    df[col2] = pd.to_datetime(df[col2])

    df['after_prev_booked'] = df.groupby(id)[col1].diff()
    df['stayed'] = df[col2] - df[col1]
    df['stayed'] = df['stayed'].shift(1)

    df = df[df['after_prev_booked'].notnull()]

    df['after_released'] = df['after_prev_booked'] - df['stayed']
    # Convert the datetime type to integer
    df['after_released'] = df['after_released'].astype('timedelta64[D]')


    for window in windows:
        within = int(window*days)
        time_frame = 'within_{}yrs'.format(int(within/365))
        df[time_frame] = np.where(df['after_released'] <= within, 1, 0)


    return df, time_frame



# GIVEN TRAIN&TEST PERIODS, SPLIT DATA INTO TRAIN/TEST SETS 
def extract_train_test_sets(df, col, train_start, train_end, test_start, test_end):

    train_set = df[(train_start <= df[col]) & (df[col]<= train_end)]
    test_set = df[(test_start <= df[col]) & (df[col]<=test_end)]

    return train_set, test_set



# TEMPORAL HOLDOUTS
def temporal_eval(features, df, col='booking_date', target=None, save=False):

    '''
    Temporal evaluation function with data-specific function(within_frame2)

    START : start date of data (2010-01-01)
    END: end date of date (2016-04-30)

    Input: target (str) - dependent column name 
           features (list) - a list of independent column names
           df (dataframe) - a entire data set 
           col (str) - a date column to be used to split the dataset

    Output: csv file
    '''

    start_time_date = datetime.strptime(START, '%Y-%m-%d')
    end_time_date = datetime.strptime(END, '%Y-%m-%d')
    
    df[col] = pd.to_datetime(df[col])

    for window in WINDOWS:
        test_end_time = end_time_date
        while (test_end_time >= start_time_date + 2 * relativedelta(months=+window)):
            test_start_time = test_end_time - relativedelta(months=+window)
            train_end_time = test_start_time  - relativedelta(days=+1) # minus 1 day
            train_start_time = train_end_time - relativedelta(months=+window)
            while (train_start_time >= start_time_date):
                print("Train_Start: {}  Train_End: {}".format(train_start_time.date(), train_end_time.date()))
                print("Test_Start: {}  Test_End: {}".format(test_start_time.date(), test_end_time.date()))
                train_start_time -= relativedelta(months=+window)
                # call function to get data
                train_set, test_set = extract_train_test_sets(df, col, train_start_time, train_end_time, test_start_time, test_end_time)
                # apply within_frame2
                train_set,target = within_frame2(train_set)
                test_set, target = within_frame2(test_set)
                # fit on train data
                x_train, x_test = train_set[features], test_set[features]
                y_train, y_test = train_set[target], test_set[target]
                # predict on test data
                #baseline = df[target].sum() / df.shape[0]
                result = classifiers_loop(x_train, x_test, y_train, y_test)
                result.to_csv('{} {} {} {}.csv'.format(train_start_time,train_end_time,test_start_time,test_end_time), mode='a', index=False)
                best = best_grid(result)
                feature_importance(x_train, y_train, best)
                #plot_mult(MODELS_TO_RUN, x_train, x_test, y_train, y_test, best)
            test_end_time -= relativedelta(months=+UPDATE)





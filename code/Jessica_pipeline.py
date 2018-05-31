from default_grids import *
from datetime import *

import sys
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.preprocessing import Imputer, StandardScaler
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
MODELS_TO_RUN = ['RF','DT','KNN','LR', 'NB']


GENERAL = True
TEMPORAL = True
ENTRY = "2011-01-01"
EXIT = "2013-12-31"
WINDOWS = [6,12]
UPDATE = 12



# READ DATA 
def read_data(file_name):
	df = pd.read_csv(file_name)

	return df


# EXPLORE DATA / VISUALIZE DATA
def explore_df(df, save=False):

	# description of a data table
	describe_df = df.describe()

	# find columns with missing values with number of missing values
	missing_df = pd.DataFrame(list(df.isnull().sum()), columns=['missing_values'], index=df.columns)

	if save:
		describe_df.to_csv('describe_df.csv')
		print('data frame description file is saved as ~.csv')

		missing_df.to_csv('missing_values.csv')
		print('data missing values file is saved as ~.csv')

	else:
		if NOTEBOOK:
			describe_df
			missing_df
		else:
			print(describe_df)
			print("\n")
			print(missing_df)

	return None


# SHOW CORRELATIONS AMONG VARS
def correlation(df, columns, save=True):

	# correlation 
	correlation_df = df.corr()

	if save:
		correlation_df.to_csv('correlation_df.csv')
		plt.figure(figsize=(correlation_df.shape[0], 5))
		corr_figure = sns.heatmap(correlation_df, xticklabels=correlation_df.columns, yticklabels=correlation_df.columns)
		print('data correlation file is saved as ~.csv')

	else:
		if NOTEBOOK:
			correlation_df
			corr_figure.figure.savefig('correlation.png')
			print('figure is saved as a file correlation.png')
		else:
			print(correlation_df)
			plt.show()



# CROSS TAB OF TWO VARS
def cross_tab(df, columns, save=True):
	if len(columns) == 2:
		cols_df = pd.crosstab(df[columns[0]], df[columns[1]])

	if save:
		cols_df.to_csv('cols_df.csv')
		print('generated crosstab with two variables')

	else:
		if NOTEBOOK:
			cols_df
		else:
			print(cols_df)

	return None


# BAR PLOT
def plot_df(df, columns, save=True):

	for col in columns:
		count_column = df[col].value_counts()
		plt.figure(figsize=(len(count_column), 5))
		column_figure = sns.barplot(count_column.index, count_column.values, alpha=0.8)
		plt.title('{} values'.format(col))
		plt.ylabel('Number of Counts', fontsize=12)
		plt.xlabel(col, fontsize=12) 
		
		if save: 
			column_figure.figure.savefig('{}.png'.format(col))
			print('figure is saved as a file ~.png')
		else:
			plt.show()

	return None



# PRINT COLUMNS WITH MISSING VALUES
def missing_vals(df, add_column=True):

	# make new columns indicating what will be imputed
	cols_with_missing = [col for col in df.columns if df[col].isnull().any()]


	for col in cols_with_missing:
		if add_column:
			df[col + '_was_missing'] = df[col].isnull()
		print('{} has missing values'.format(col))

	return cols_with_missing


# FIND OUTLIERS - BOX PLOT / OUTLIER VALUES WITH QUANTILE= threshold
def outliers_info(df, columns, threshold = 0.999, save=True):

	plt.figure(figsize=(len(columns), 5))
	outlier_fig = sns.boxplot(data=df[columns], orient='h')
	plt.title('Outliers plot')
	plt.ylabel('Variables', fontsize=12)
	plt.xlabel('Counts', fontsize=12)

	# outliers
	outlier_vals = df[columns].quantile(threshold)


	if save:
		outlier_fig.figure.savefig('outliers.png')
		print('figure is saved as a file outliers.png')

		outlier_vals.to_csv('outlier_values.csv')
		print('outlier values file is saved as ~.csv')
	else:
		plt.show()
		if NOTEBOOK:
			outlier_vals
		else:
			print(outlier_vals)

	return None



# PREPROCESSING #


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



# FILTER WITH FREQUENCY OF RE-ENTRY
def within_frame(df, id_num='mni_no', timestamp='release_date_y', col_name='re-enter-days', duration=365):
    df = df.sort_values(by=[id_num, timestamp])
    df['{}'.format(col_name)] = df.groupby(id_num)[timestamp].diff()
    df['{}'.format(col_name)] = df['{}'.format(col_name)].apply(lambda x: x.days)
    df['within_{}'.format(duration)] = np.where(df['{}'.format(col_name)]>duration, 1, 0)

    
    return df



# FILL IN MISSING VALUES WITH SELECTED METHODS FOR EACH TYPE 
def fill_missing(df, method="mean"):

	for col in df:
		if df[col].dtype == 'object':
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


# IMPUTE MISSING VALUES USING IMPUTER
def imputer(df):

	# keep int columns
	int_columns = [col for col in df.columns if df[col].dtypes=='int']

	# make copy of the original data
	copy_df = df.copy()

	# imputation
	imputer = Imputer()
	imputed_df = pd.DataFrame(imputer.fit_transform(copy_df))
	imputed_df.columns = copy_df.columns


	return imputed_df, int_columns


# CONVERT TYPE OF GIVEN COLUMNS TO SELECTED TYPE
def type_to(df, cols, conv_type="int"):

	if conv_type=="int":
		for col in cols:
			df[col] = df[col].astype(int)

	if conv_type=="float":
		for col in cols:
			df[col] = df[col].astype(float)

	if conv_type=="str":
		for col in cols:
			df[col] = df[col].astype(str)

	print('given columns {} are coverted to {} type'.format(cols,conv_type))

	return df


# CONVERT GIVEN VALS(val1, val2) TO BINARY VALS(0,1) 
def special_convert(df, cols, val1="t", val2="f"):
	for col in cols:
		df[col] = df[col].map({val1: 1, val2: 0})

	return df
    


# ROUND VALUES OF GIVEN COLUMNS WITH SPECIFIED PLACE
def round_to(df, cols=None, digit=0):

	if cols:
		cols = cols
	else:
		cols = df.columns

	for col in cols:
		df[cols].round(digit)

	print('given columns are rounded to {}'.format(digit))

	return df


# FIND OUTLIERS
def iqr_outliers(col):
	q1, q3 = np.percentile(col, [25,75])
	iqr = q3-q1
	lower_bound = q1 - (iqr*1.5)
	upper_bound = q3 + (iqr*1.5)

	return np.where((col>upper_bound) | (col <lower_bound))



# SET THE CEILINGS & BUCKETING
def ceiling(x, cap_max):
	if x > cap_max:
		return cap_max
	else:
		return x

def flooring(x, cap_min):
	if x < cap_min:
		return cap_min
	else:
		return x

# SET BOUNDARIES WITH GIVEN UPPER AND LOWER CAPPING VALUES
def set_boundaries(df,columns,upper,lower):

	for col in columns:
		upper_outlier = df[col].quantile(upper)
		df[col] = df[col].apply(lambda x: ceiling(x, upper_outlier))
		lower_outlier = df[col].quantile(lower)
		df[col] = df[col].apply(lambda x: flooring(x, lower_outlier))

	return None


# DISCRETIZING COLUMNS
def into_buckets(df, columns, quant=10, dup='drop'):
    # q=4 for quartiles, q=10 for deciles
    created_buckets = []

    for col in columns:

        if type(col) == 'int':
        	quant = np.linspace(0,1,quant+1)
        	bins = algos.quantile(col, quant)
        	bins = np.unique(bins)
        	df[col+'_buckets'] = pd.cut(df[col], bins=bins, include_lowest=True)
        else:
        	df[col+'_buckets'] = pd.qcut(df[col], q=quant, duplicates=dup)

        created_buckets.append(col+'_buckets')

    return df, created_buckets



# CREATE DUMMIES
def create_dummy(df, columns):
	for col in columns:
		dummy = pd.get_dummies(df[col], prefix=col)
		df = pd.concat([df, dummy], axis=1)

	return df



# CHANGE IN VARS FOR EACH
def is_changed(df, id_num='mni_no', var='mar_stat'):
	df['{}_changed'.format(var)] = (df[var]!=df[var].shift()) | (df[id_num]!=df[id_num].shift())

	return df
	


# CUM SUM OF VARS
def cum_vals(df, id_num='mni_no', var='re-enter-days'):

	df = df.sort_values(by=[id_num, var])
	df['{}_so_far'.format(var)] = df.groupby(id_num)[var].cumsum()

	return df



# BUILD CLASSIFIER & EVALUATION #

# SPLIT DATA INTO TRAIN/TEST SETS
def split_data(df, target, features, test_size=0.2):
	X = df[features]
	Y = df[target]
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = 42)

	return x_train, x_test, y_train, y_test


# CHOOSE CLASSIFIERS TO BE USED AND FIT DATA
def classifier(model, num, x_train, y_train):
	if model == 'DT':
		clf = DecisionTreeClassifier(criterion='entropy', max_depth=num)
	elif model == 'LR':
		clf = LogisticRegression(penalty='l2')
	elif model == 'RF':
		clf = RandomForestClassifier(max_depth=num)
	elif model == 'KNN':
		clf = KNeighborsClassifier(n_neighbors=num)


	fitted_clf = clf.fit(x_train, y_train)

	return fitted_clf


# PRINT OUT SENSIBLE SCORES OF A MODEL
def evaluation(fitted_clf, x_test, y_test, save=False):
	y_pred = fitted_clf.predict(x_test)

	report = classification_report(y_test, y_pred)
	score = fitted_clf.score(X=x_test, y=y_test)
	MAE = metrics.mean_absolute_error(y_test, y_pred)
	MSE = metrics.mean_squared_error(y_test, y_pred)
	RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
	accuracy = accuracy_score(y_test, y_pred)

	if save:
		with open("Classifier_result.txt", "w") as output_file:
			output_file.write("Classification Report: \n{} \n\nClassifier Score: {} \nAbsolute Error: {} \nSquared Error: {} \nMean Squared Error: {}\
				\nAccuracy Score: {}".format(report, score, MAE, MSE, RMSE, accuracy))
		print('Result is saved as ~.txt')
	else:
		print(report, score, MAE, MSE, RMSE, accuracy)



# based on Rayid's magicloops from here
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



# FEATURE IMPORTANCES 
def feature_importance(clf, model, save=False):

	if FEATURE_CLFS[model] == 'feature_importances':
		importances = clf.feature_importances_

	if FEATURE_CLFS[model] == 'coef':
		importances = clf.coef.tolist()


	data = list(importances)
	features_df = pd.DataFrame(data, columns=['importance'], index=X.columns)
	sorted_features = features_df.sort_values(by='importance', ascending=0)

	plt.figure(figsize=(15,6))
	top15 = sorted_features.head(15)
	features_figure = sns.barplot(top15.index, top15.values.flatten(), alpha=0.8)
	plt.title('{} Importance of Features'.format(model))
	plt.ylabel('Importance Value', fontsize=12)
	plt.xlabel('Features', fontsize=12)
	plt.xticks(rotation = 90)


	if save:
		sorted_features.to_csv('{}.csv'.format(model))
		print("List of features is saved as ~.csv")

		features_figure.figure.savefig('{}.png'.format(model))
		print('Figure is saved as a file {}_features.png'.format(model))
	else:
		print(sorted_features)
		plt.show()

	return None


# TEMPORAL HOLDOUTS
def temporal_eval(target, features, df, col, save=True):

	start_time_date = datetime.strptime(START, '%Y-%m-%d')
	end_time_date = datetime.strptime(END, '%Y-%m-%d')

	for window in WINDOWS:
		test_end_time = end_time_date
		while (test_end_time >= start_time_date + 2 * relativedelta(months=+window)):
			test_start_time = test_end_time - relativedelta(months=+window)
			train_end_time = test_start_time  - relativedelta(days=+1) # minus 1 day
			train_start_time = train_end_time - relativedelta(months=+window)
			while (train_start_time >= start_time_date):
				print (train_start_time,train_end_time,test_start_time,test_end_time, window)
				train_start_time -= relativedelta(months=+window)
				# call function to get data
				train_set, test_set = extract_train_test_sets(df, col, train_start_time, train_end_time, test_start_time, test_end_time)
				# fit on train data
				x_train, x_test = train_set[features], test_set[features]
				y_train, y_test = train_set[target], test_set[target]
				# predict on test data
				result = classifiers_loop(x_train, x_test, y_train, y_test)
				result.to_csv('{} {} {} {}.csv'.format(train_start_time,train_end_time,test_start_time,test_end_time), mode='a', index=False)
			test_end_time -= relativedelta(months=+UPDATE)


# GIVEN TRAIN&TEST PERIODS, SPLIT DATA INTO TRAIN/TEST SETS 
def extract_train_test_sets(df, col, train_start, train_end, test_start, test_end):

	train_set = df[(train_start <= df[col]) & (df[col]<= train_end)]
	test_set = df[(test_start <= df[col]) & (df[col]<=test_end)]

	return train_set, test_set

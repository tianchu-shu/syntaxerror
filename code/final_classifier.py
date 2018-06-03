#Modified based on Rayid's magic loop
from final_default_grids import * 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import *
import random
import pylab as pl
import matplotlib.pyplot as plt
import time
import seaborn as sns
import datetime



MODELS_TO_RUN = ['RF', 'ET', 'AB', 'LR', 'SVM', 'GB', 'NB', 'DT', 'SGD', 'KNN']

CLFS = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }



FEATURE_CLFS = {'RandomForestClassifier':'feature_importances',
        'ExtraTreesClassifier': 'feature_importances',
        'AdaBoostClassifier': 'feature_importances',
        'LogisticRegression': 'coef',
        'svm.SVC': 'coef',
        'GradientBoostingClassifier': 'feature_importances',
        'GaussianNB': None,
        'DecisionTreeClassifier': 'feature_importances',
        'SGDClassifier': 'coef',
        'KNeighborsClassifier': None,
        'linear.SVC': 'coef'}

            
LARGE_GRID = { 
'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
'NB' : {},
'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
       }

SMALL_GRID = { 
'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
'NB' : {},
'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
       }

TEST_GRID = { 
'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
'LR': { 'penalty': ['l1'], 'C': [0.01]},
'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
'NB' : {},
'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
'SVM' :{'C' :[0.01],'kernel':['linear']},
'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
       }


def generate_binary_at_k(y_scores, k):
    '''
    Set first k% as 1, the rest as 0.
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def scores_at_k(y_true, y_scores, k):
    '''
    For a given level of k, calculate corresponding
    precision, recall, and f1 scores.
    '''
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    recall = recall_score(y_true, preds_at_k)
    f1 = f1_score(y_true, preds_at_k)
    return precision, recall, f1


def plot_precision_recall(y_true, y_prob, model, p):
    '''
    Plots the PR curve given true value and predicted
    probilities of y.
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)    
    plt.clf()
    plt.plot(recall, precision, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve for {} model: AUC={:.2f} \n with parameters: {}'.\
        format(model, average_precision_score(y_true, y_prob), p))
    plt.legend(loc="lower left")
    plt.show()


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

def extract_train_test_sets(df, col, train_start, train_end, test_start, test_end):

	train_set = df[(train_start <= df[col]) & (df[col]<= train_end)]
	test_set = df[(test_start <= df[col]) & (df[col]<=test_end)]

	return train_set, test_set


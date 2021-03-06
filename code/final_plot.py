import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import seaborn as sns; sns.set()
from sklearn.linear_model import SGDClassifier 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import *
import time
import seaborn as sns


clfs = {'RF': RandomForestClassifier(),
    'Boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)),
    'Logit': LogisticRegression(),
    'SVM': SVC(probability=True, random_state=0),
    'Tree': DecisionTreeClassifier(),
    'Bag': BaggingClassifier(),
    'KNN': KNeighborsClassifier(),
    'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
    'NB': GaussianNB()}


def bar_charts(results):
    for item in ['auc-roc','precision','time']:
        plt.figure()
        results.groupby(['model_type'])[item].mean().plot(kind='bar', title='Average '+item+' across classifiers')
        plt.ylabel(item)


def feature_importance(x_train, y_train, bestm, x="ET", k=10):
    '''
    Based on the best grid for each classifer, print out the 
    top k important features
    '''
    clf = clfs[x]
    for p in ParameterGrid(bestm[x]):
        clf.set_params(**p)
    forest = clf.fit(x_train, y_train)
    indepv = list(x_train.columns)
    importances = forest.feature_importances_
    current_palette = sns.color_palette(sns.color_palette("cubehelix", k))
    
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
    for index, clf in enumerate([clfs[x] for x in models]):
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

    
def plot_df(df, columns, save=False):
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

    
#Printing out the best decision tree
def print_tree(x_train, y_train, bestm):
    '''
    Based on the best grid for, print out the tree graph    
    '''
    clf = clfs["Tree"]
    for p in ParameterGrid(bestm["Tree"]):
        clf.set_params(**p)
        tree = clf.fit(x_train, y_train)
        indepv= list(x_train.columns)
        tree_viz = export_graphviz(tree, out_file=None, feature_names=indepv, rounded=True, filled=True)
        graph =graphviz.Source(tree_viz)
            
    return graph



def plot_best(models, x_train, x_test, y_train, y_test, bestm):
    '''
    Run model with the best given params on x and y
    and print out the scores for comparison
    '''
    for index, clf in enumerate([clfs[x] for x in models]):
        model_params = bestm[models[index]]
        for p in ParameterGrid(model_params):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(x_train, y_train).predict_proba(x_test)[:,1]
                plot_precision_recall_n(y_test, y_pred_probs, models[index],p)
                
            except IndexError as e:
                print(e)
                continue



def plot_precision_recall_n (y_true, y_score, model_name,  para = None, fig =None):
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
    ax1.plot(pct_above_per_thresh, precision_curve, "#3498db")
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color="#3498db")
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, '#e74c3c')
    ax2.set_ylabel('recall', color='#e74c3c')
    
    plt.title('2-class Precision-Recall curve for {} model: AUC={:.2f} \n with parameters: {}'.\
                        format(model_name, average_precision_score(y_true, y_score), para))
    plt.savefig(model_name)
    plt.show()

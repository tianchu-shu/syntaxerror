import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



# Build a forest and compute the feature importances
def feature_importance(x_train, y_train，k=10):
    forest = ExtraTreesClassifier(n_estimators=100, max_depth=5, criterion='entropy', min_samples_split=10, n_jobs=-1,
                                random_state=0)

    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    current_palette = sns.color_palette(sns.color_palette("hls", 8))
    indices = np.argsort(importances)[::-1]
    indices = indices[:k]

    # Print the feature ranking
    print("Feature ranking:")

    labels_arr = []
    for f in range(len(indices)):
        label = indepv[indices[f]]
        labels_arr.append(label)
        print("%d. %s (%f)" % (f+1, label, importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], align="center", tick_label = labels_arr, color= current_palette)
    plt.xlim([-1, len(indices)])
    plt.xticks(range(len(indices)),labels_arr, rotation = 'vertical')
    plt.show()
  
  

def plot_mult(models, x_train, x_test, y_train, y_test, bestm):
    '''
    Run model with the best given params on x and y
    and print out all the best models' on the same graph
    '''
    colors = "bgrcmykw"
    color_index = 0

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('recall', color='r')
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
                
                ax1.plot(pct_above_per_thresh, precision_curve, c=colors[color_index])
                ax2.plot(pct_above_per_thresh, recall_curve, c=colors[color_index])                
            except IndexError as e:
                print(e)
                continue
        color_index += 1
        if color_index >= 7:
            color_index = 0
    ax1.legend(models)
    plt.show()

    
 #Printing out the best decision tree
def print_tree(x_train, y_train, bestm, models=["Tree"]):
    for index, clf in enumerate([clfs[x] for x in models]):
        model_params = bestm[models[index]]
        for p in ParameterGrid(model_params):
            clf.set_params(**p)
            tree = clf.fit(x_train, y_train)
            tree_viz = export_graphviz(tree, out_file=None, feature_names=indepv, rounded=True, filled=True)
            graph =graphviz.Source(tree_viz)
            
    return graph

GRID = SMALL_GRID
MODELS_TO_RUN = ['RF','DT','KNN','LR', 'NB']


GENERAL = False
TEMPORAL = True
START = "2011-01-01"
END = "2013-12-31"
WINDOWS = [6,12]
UPDATE = 12



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
				result.to_csv('{} {} {} {}.csv'.format(train_start_time,train_end_time,test_start_time,test_end_time), index=False)
			test_end_time -= relativedelta(months=+UPDATE)


# GIVEN TRAIN&TEST PERIODS, SPLIT DATA INTO TRAIN/TEST SETS 
def extract_train_test_sets(df, col, train_start, train_end, test_start, test_end):

	train_set = df[(train_start <= df[col]) & (df[col]<= train_end)]
	test_set = df[(test_start <= df[col]) & (df[col]<=test_end)]

	return train_set, test_set
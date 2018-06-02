def plot_best(models, x_train, x_test, y_train, y_test, bestm = best):
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
				
def plot_precision_recall_n(y_true, y_score, model_name,  para = None, fig =None):
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
    
    plt.title('2-class Precision-Recall curve for {} model: AUC={:.2f} \n with parameters: {}'.\
                        format(model_name, average_precision_score(y_true, y_score), para))
    plt.show()
	
def plot_mult(models, x_train, x_test, y_train, y_test, bestm = best):
    '''
    Run model with the best given params on x and y
    and print out the scores for comparison
    '''
    for index, clf in enumerate([clfs[x] for x in models]):
		colors = "bgrcmykw"
		color_index = 0
		
        model_params = bestm[models[index]]
		fig, ax1 = plt.subplots()
		ax1.set_xlabel('percent of population')
		ax1.set_ylabel('precision', color='b')
		ax2.set_ylabel('recall', color='r')
        for p in ParameterGrid(model_params):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(x_train, y_train).predict_proba(x_test)[:,1]
				
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
                
				ax1.plot(pct_above_per_thresh, precision_curve, c=colors[color_index])
				color_index += 1
				ax2.plot(pct_above_per_thresh, recall_curve, c=colors[color_index])
				if color_index == 7:
					color_index = 0
				
				
            except IndexError as e:
                print(e)
                continue
		plt.show()
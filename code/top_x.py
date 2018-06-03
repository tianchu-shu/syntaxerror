models = []
for row in best:
    models.append(row)


def finding_risk_scores(models, x_train, x_test, y_train, y_test, grid=best):
    '''
    Adds the y-pred probs for each model to the x_test
    Can be used to find top X% of people at risk according to any given model
    '''
    #new_df = pd.DataFrame(columns = ('Bag', 'Boost', 'ET', 'KNN', 'Logit', 'NB', 'RF', 'Tree'))
    results_df = pd.DataFrame(columns=('model_type', 'parameters', 'auc-roc', 'precision', 'time', 'precision,recall,f1 at_5',
     'precision,recall,f1 at_10', 'precision,recall,f1 at_20', 'precision,recall,f1 at_30', 'precision,recall,f1 at_50'))
    x_test_copy = x_test.copy()
    for index, clf in enumerate([clfs[x] for x in models]):
        model_params = grid[models[index]]
        for p in ParameterGrid(model_params):
                clf.set_params(**p)
                y_pred_probs = clf.fit(x_train, y_train).predict_proba(x_test)[:,1]
                name = str(models[index])
                print (name)
                x_test_copy[name] = y_pred_probs
                
    return x_test_copy
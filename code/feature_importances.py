import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from datetime import datetime

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
	plt.title('Decision Tree Importance of Features')
	plt.ylabel('Importance Value', fontsize=12)
	plt.xlabel('Features', fontsize=12)
	plt.xticks(rotation = 90)


	if save:
		sorted_features.to_csv('decisiontree_classifer.csv')
		print("list of features is saved as ~.csv")

		features_figure.figure.savefig('Decisiontree_features.png')
		print('figure is saved as a file Decisiontree_features.png')
	else:
		print(sorted_features)
		plt.show()

	return None
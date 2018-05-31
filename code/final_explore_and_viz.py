'''
Functions for data exploration and vizualization
'''


def basic_exploring (dataframe):
    '''
    Given a dataframe, the function does some basic exploring
        by printing the description and information of the dataset
        
    input:
        dataframe
    '''
    print ('DESCRIBING DATASET: \n \n', dataframe.describe(), '\n \n') 
    print ('DATASET INFORMATION \n') 
    print (df.info(), '\n \n \n')
    print ('DATASET HEAD \n')
    print (df.head(), '\n \n \n')
    print ('DATASET TAIL \n')
    print (df.tail(), '\n \n \n')
    
    for column in df.columns:
        plotting_curves (df, column)
        plt.show()
        
        
        
  
def corr_matrix(df):
    '''
    Creates a heatmap that shows the correlations between the different variables in a dataframe.
    
    Input:
        df: a dataframe
        title: name of the correlation_matrix
        
    Return:
        Outputs a heatmatrix showing correlations
    
    
    '''
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, 
                mask=np.zeros_like(corr, dtype=np.bool), 
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

    
    
def plotting_curves (dataframe, feature):
    '''
    Given a dataframe, a column name, 
        displays a plot of that dataframe column distribution.
        
    Input:
        dataframe
        feature: column name (string)
        
    Return:
        displays a distribution of that variable
        
    Inspired by:
        https://seaborn.pydata.org/generated/seaborn.distplot.html
    '''
    title = feature + ' Graph'
    ax = sns.distplot(dataframe[feature])
    ax.set_title(title)
    
    def create_graph(df, variable, subject_variable, type = 'mean', graph_type = 'line'):
	'''
	Take a variable and create a line chart mapping that variable
	against a dependent_variable, serious delinquency in the prior two years
	Inputs:
	df: A panda dataframe
	variable: A string, which is a column in df
	Outputs:
	Variable_chart: A matplotlib object of the resultant chart
	'''
	columns = [subject_variable, variable]
	if type == 'mean':
		var_plot = df[columns].groupby(subject_variable).mean()
	elif type == 'total':
		var_plot = df[columns].groupby(subject_variable).sum()
	
	graph = var_plot.plot(kind = graph_type, use_index = False, figsize = (10,5))

	return graph


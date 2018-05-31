'''
Collection of pre-processing functions for ML Project
31st May 2018
'''




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
    
    
    
def fill_missing(df, list_to_fill, operation_type, value = None):
	'''
	Fill in null values with the mean, median or a set value of the column.
	Input:
	df: A panda dataframe
	list_to_mean: List of columns to act on
	Outputs:
	df
	'''
	
	for variable in list_to_mean:
		if operation_type == 'mean':
			df[variable].fillna(df[variable].mean(), inplace=True)
		elif operation_type == 'median':
			df[variable].fillna(df[variable].median(), inplace=True)
        elif operation_type == 'mode':
			df[variable].fillna(df[variable].mode(), inplace=True)
		elif operation_type == 'set':
			df[variable].fillna(value, inplace=True)
	
	return df

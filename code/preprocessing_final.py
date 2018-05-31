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


def outlier(df, variable):
	'''
	Locate outliers in given column and eliminate them.
	Inputs:
	df: A pandas dataframe
	variable: target column name, string
	Outputs:
	changed_df: A pandas dataframe without the outliers
	'''
	low_out = df[variable].quantile(0.005)
	high_out = df[variable].quantile(0.995)
	df_changed = df.loc[(df[variable] > low_out) & (df[variable] < high_out)]

	number_removed = df.shape[0] - df_changed.shape[0]
	print("Removed" + str(number_removed) + " outliers from " + variable)

	return df_changed




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

def processing_drop(df, drop_list, target_quantifier, value):
	'''
	1) Drops all rows where the variables in the drop_list value where the target is less than, greater than, or equal to a value.
	Input:
	df: A panda dataframe
	drop_list: List of columns to act on
	maximum: The integer to drop if the value is greater than
	Outputs:
	df
	'''
	for variable in drop_list:
		if target_quantifier == 'equal':
			df = df[df[variable] == value]
		elif target_quantifier == 'greater':
			df = df[df[variable] >= value]
		elif target_quantifier == 'lesser':
			df = df[df[variable] <= value]
	
	return df


# ROUND VALUES OF GIVEN COLUMNS WITH SPECIFIED PLACE
def round_to(df, cols=None, digit=0):

	if not cols:
		cols = df.columns

	for col in cols:
		df[cols].round(digit)

	print('given columns are rounded to {}'.format(digit))

	return df
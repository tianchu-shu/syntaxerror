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
    
    
    
    
    
def  Fill_in(df,  cols, method="mean"):
         ''''''
     Filling in missing values with "mean" or "median"  
    
    Inputs:
        df: (pandas dataframe)
        a list of those column names
        method (string): mean or median
 
    Returns:
        pandas dataframe
    '''
    for col in cols:
        if method =='mean':
            val = df[col].mean()
        elif method =='median':
            val = df[col].median()
        elif method =='mode':
            val = df[col].mode()
            val = str(val)
        elif method == 'missing':
            val = "missing"
        df[col] = df[col].fillna(val)
        print ('Filling missing value for {} using {}'.format(col, method))
    return df

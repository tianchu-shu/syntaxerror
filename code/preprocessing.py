'''
Amir Akhter Kazi
Checks for nulls, and fills in missing data according to a criteria
Also focuses on outliers and removes (Though not sure this works in all cases)

25th May 2018
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


def fill_in_missing_data (dataframe, criteria):
    '''
    Given a dataframe and a criteria (options: mean, median or mode),
        fills in the NaN or Null values in that column for the dataframe
        based on the given criteria
        
    Input:
        dataframe
        criteria: string
    '''
    features_with_nulls = checking_for_nulls(dataframe)
    
    for feature in features_with_nulls:
        if criteria == 'mean': input_value = df[feature].mean() 
        if criteria == 'median': input_value = df[feature].median() 
        if criteria == 'mode': input_value = float(df[feature].mode())
        
        dataframe[feature] = dataframe[feature].fillna(input_value)
            


### Not sure if below outlier functions work in all cases
### Check again.

def removing_outliers (dataframe):
    '''
    For a given dataframe, loops over the features 
        and removes any rows which have an outlier for any given feature
    Uses the removing_outliers_helper function in its task
    
    input: dataframe
    
    output: dataframe (with outliers removed)
    '''
    df = dataframe.columns
    
    for feature in df:
        dataframe = removing_outliers_helper(dataframe, feature)
        
    return dataframe

def removing_outliers_helper (dataframe, feature, upper_quartile = 0.995, lower_quartile = 0.005):
    '''
    For a given dataframe and feature,
        removes those rows which are outliers (i.e. above the 99.5 percentile, or below the 0.5 percentile)
        
    Returns the dataframe after removing those rows
    
    Input: 
        dataframe
        feature: string
        
    Output:
        returns dataframe (with outliers of a feature removed)
        
    Inspired by:
    https://stackoverflow.com/questions/35827863/remove-outliers-in-pandas-dataframe-using-percentiles/35828995
    '''

    upper_bound = dataframe[feature].quantile(upper_quartile)
    lower_bound = dataframe[feature].quantile(lower_quartile)
    dataframe = dataframe.loc[(dataframe[feature] >= lower_bound) & (dataframe[feature] <= upper_bound)]

    return dataframe


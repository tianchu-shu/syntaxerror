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

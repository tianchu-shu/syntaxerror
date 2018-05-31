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
        
        
        
  

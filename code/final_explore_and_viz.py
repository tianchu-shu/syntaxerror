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


def splitting_across_variable(dataframe, feature):
    '''
    For a given dataframe and feature,
        it splits the database as per the values of the feature,
        and gives aggregrate information for that feature and
        how it relates to other features
        
    Input:
        dataframe
        
    Output: 
        dataframe
    '''
    return dataframe.groupby([feature]).mean().transpose()


def value_counter (dataframe, feature):
    '''
    Given a dataframe, and feature,
        gives the number of occurrences of each value in that feature column
        
    Input:
        dataframe
        feature: string
    
    Output: dataframe
    '''
    return dataframe[feature].value_counts()


def comparing_two_features(dataframe, feature_1, feature_2):
    '''
    Given a dataframe, and two features,
        does a cross tab and shows values across those two corresponding features
        
    Input:
        dataframe
        feature_1, feature_2: string
        
    Output: 
        dataframe
        
    '''
    return pd.crosstab(dataframe[feature_1], dataframe[feature_2])


def plotting_two_feature_comparison(dataframe, x_feature, y_feature):
    '''
    Given a dataframe and two features,
        plots a graph across those two features
        
    Input:
        dataframe
        x_feature, y_feature: string
        
    Output:
        matplotlib.axes._subplots.AxesSubplot
    '''
    return dataframe[[x_feature, y_feature]].groupby(x_feature).sum().plot()



# BAR PLOT
def plot_df(df, columns, save=True):

    for col in columns:
        count_column = df[col].value_counts()
        plt.figure(figsize=(len(count_column), 5))
        column_figure = sns.barplot(count_column.index, count_column.values, alpha=0.8)
        plt.title('{} values'.format(col))
        plt.ylabel('Number of Counts', fontsize=12)
        plt.xlabel(col, fontsize=12) 
        
        if save: 
            column_figure.figure.savefig('{}.png'.format(col))
            print('figure is saved as a file ~.png')
        else:
            plt.show()

    return None

def counting_uniques(df):
    '''
    For the given dataframe, gives a sum of the unique values in each feature.
    Then prints a plot bar to represent that.
    '''
    print (df.nunique())
    print (df.nunique().plot.bar())


def counting_in_a_variable(feature, df):
    '''
    For a given feature and dataframe,
        prints the total count of each category of that feature.
    '''
    print (df[feature].value_counts())

    
def making_pie (df, feature):
    '''
    Gives a pie plot of data in any feature
    '''
    if (df[feature].nunique()) < 10:    
        try: 
            df.groupby([feature]).size().plot(kind='pie')
            plt.show()
        except:
            pass
    else:
        print (' ')

    

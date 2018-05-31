import pandas as pd

# CONVERT INTO DATETIME FORMAT
def to_datetime(df, cols, index=-8, format="%Y%m%d"):
    for col in cols:
        df[col] = df[col].apply(lambda x: str(x)[:index])
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = pd.to_datetime(df[col], format=format)
        
    return df



# CUTTING & MERGING DATAFRAME WITH DATETIME RANGE 
def restrain_datetime(df, date_col='arrest_date', from_date=(2010,1,1), to_date=(2015,12,31)):
	df.index = df[date_col]
    df = df.sort_index()
	df = df[datetime(*from_date):datetime(*to_date)]
	df = df.reset_index(drop=True)

	return df



# FILTER WITH FREQUENCY OF RE-ENTRY
def within_frame(df, id_num='mni_no', timestamp='release_date_y', col_name='re-enter-days', duration=365):
    df = df.sort_values(by=[id_num, timestamp])
    df['{}'.format(col_name)] = df.groupby(id_num)[timestamp].diff()
    df['{}'.format(col_name)] = df['{}'.format(col_name)].apply(lambda x: x.days)
    df['within_{}'.format(duration)] = np.where(df['{}'.format(col_name)]>duration, 1, 0)

    
    return df

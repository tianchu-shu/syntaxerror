import pandas as pd
import numpy as np

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



# FILTER WITH FREQUENCY OF RE-ENTRY (PREV BOOKING DATE ~ NEXT BOOKING DATE)
def within_frame(df, id_num='dedupe_id', timestamp='booking_date', col_name='re-entry', duration=365):
	df = df.sort_values(by=[id_num, timestamp])
	df['{}'.format(col_name)] = df.groupby[id_num][timestamp].diff()
	df['{}'.format(col_name)] = df['{}'.format(col_name)].apply(lambda x: x.days)
	df['within_{}'.format(duration)] = np.where(df['{}'.format(col_name)]>duration, 1, 0)    
	
	return df

# FILTER WITH FREQUENCY OF RE-ENTRY (PREV RELEASED DATE ~ NEXT BOOKING DATE)
def within_frame2(df, id='dedupe_id', col1='booking_date', col2='release_date', days=365, windows=[0.5,1,2]):
	
	# Change to datetime
	df[col1] = pd.to_datetime(df[col1])
	df[col2] = pd.to_datetime(df[col2])

	df['after_prev_booked'] = df.groupby(id)[date_col].diff()
	df['stayed'] = df[col2] - df[col1]

	df = df[df['after_prev_booked'].notnull()]

	df['after_released'] = df['after_prev_booked'] - df['stayed']
	# Convert the datetime type to integer
	df['after_released'] = df['after_released'].astype('timedelta64[D]')


	for window in windows:
		within = int(window*days)
		df['within_{}days'.format(within)] = np.where(df['after_released'] <= within, 1, 0)


	return df


def temporal_split(df, time_col, start_time, mid_time, end_time):
    train = df[(df[time_col] >= start_time) & (df[time_col] < mid_time) ]
    test = df[(df[time_col] >= mid_time)  & (df[time_col] < end_time)]
    train = train.drop([time_col], axis=1)
    test = test.drop([time_col], axis=1)
    return train, test


def split_data(train, test, y):
    '''
    Split the data into training and testing set
    
    And save them to run try different models
    '''
    x_test = test[indepv] 
    x_train = train[indepv]
    y_test = test[y]
    y_train = train[y]
    
    return x_train, x_test, y_train, y_test

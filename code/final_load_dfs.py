import pandas as pd
import final_connection

def load_from_file(dataframe_string, row_num = None):
	'''
	Function reads the .csv file into a panda dataframe.
	Inputs:
	dataframe_string: String for file name
	filetype: string for file_type
	row_num: integer, number of rows to read if excel or csv
	Outputs:
	df = A panda dataframe
	'''
	if '.csv' in dataframe_string:
		df = pd.read_csv(dataframe_string, nrows = row_num)
	elif '.xls' in dataframe_string:
		df = pd.read_excel(dataframe_string, nrows = row_num)
	elif '.json' in dataframe_string:
		df = pd.read_json(dataframe_string)
	
	print("Loaded" + dataframe_string)
	
	return df

def load_from_db(table_list):
	'''
	Utilize the connection function from final_connection to return a dictionary
	of tables, converted to pandas DFs.
	'''
	df_dict = {}
	conn = final_connection.Connect()
	for x in table_list:
		file_name = x + "_df"
		df_dict[file_name] = connection.return_df('table', x)
		print("Loaded" + file_name)
	
	return df_dict
	
	

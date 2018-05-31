import connection

def load_df(table_list):
	df_dict = {}
	conn = connection.Connect()
	for x in table_list:
		file_name = x + "_df"
		df_dict[file_name]= connection.print_df('table', x)
	
	return df_dict
	
	

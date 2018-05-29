from api import *

import pandas as pd
import numpy as np


def run(data, zipcodes):

	# Getting geo info with zipcodes
	data = to_lat_lon(data, zipcodes)
	get_fips(data)
	fips = break_down(data)
	asc = info_retrieve(fips)


	# Dropping duplicated columns
	asc_cols = list(unique_asc.columns)
	df_cols = list(total_df.columns)
	cols = list(set(asc_cols).difference(set(df_cols)))
	asc = asc.drop(cols+['state'], axis=1)

	# Merging with asc data
	total_df = merge(fips, asc)

	return total_df



if __name__=="__main__":

	data = pd.read_csv("df.csv")
	zipcodes = pd.read_csv("zipcodes.csv")
	run(data, zipcodes)

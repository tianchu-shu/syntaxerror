from final_pipeline import *
from final_plot import *
from jocojims import *



def run():


	# READ DATA
	df = load_from_file('final_data.csv')
	acs = load_from_file('fips_acs.csv')

	# CLEAN & MERGE DATA
	acs = acs.drop(columns=ACS_DROP, axis=1)
	df = df.dropna(subset=['zip'])

	df = acs.merge(df, how="inner")
	df = df.drop_duplicates()

	df = df.drop(DROP_COLS, axis=1)

	# PRE-PROCESS DATA
	df, binned = bin_gen(df, CATS, label='binned', fix_value='prefix')
	df = dummy_variable(df, DUMMIES+binned)

	remove = DATE_COLS
	features = binned+DUMMIES
	features = list(set(features).difference(set(remove)))

	# MODELS & EVALUATION
	result = temporal_eval(features, df)



if __name__=="__main__":
	run()